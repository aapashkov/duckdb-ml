// Copyright 2026
//
// ml_predict table function for PCA projection.
//
// Algorithm: sklearn-like transform using stored PCA model.
// Library: Eigen.
// Constraints: output row-count equals input row-count, prediction columns only.

#include "ml/register.hpp"

#include "ml/boosted_tree_regressor.hpp"
#include "ml/pca.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {
namespace ml {
namespace {

struct MlPredictBindData : public FunctionData {
	MlPredictBindData(string model_type_p, PcaModel pca_model_p, BoostedTreeRegressorModel boosted_model_p,
	                 vector<idx_t> feature_indices_p)
	    : model_type(std::move(model_type_p)), pca_model(std::move(pca_model_p)),
	      boosted_model(std::move(boosted_model_p)), feature_indices(std::move(feature_indices_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MlPredictBindData>(model_type, pca_model, boosted_model, feature_indices);
	}

	bool Equals(const FunctionData &other) const override {
		auto &rhs = other.Cast<MlPredictBindData>();
		return model_type == rhs.model_type && feature_indices == rhs.feature_indices;
	}

	string model_type;
	PcaModel pca_model;
	BoostedTreeRegressorModel boosted_model;
	vector<idx_t> feature_indices;
};

struct MlPredictGlobalState : public GlobalTableFunctionState {
	explicit MlPredictGlobalState(const MlPredictBindData &bind_data) {
		if (bind_data.model_type == "boosted_tree_regressor") {
			predictor = make_uniq<BoostedTreePredictor>(bind_data.boosted_model);
		}
	}

	unique_ptr<BoostedTreePredictor> predictor;
};

static vector<idx_t> ResolveFeatureIndices(const PcaModel &model, const vector<string> &input_names) {
	vector<idx_t> indices;
	indices.reserve(model.n_features);
	if (model.feature_names.empty()) {
		if (input_names.size() < model.n_features) {
			throw InvalidInputException("ml_predict input has %llu columns, but model expects %llu features",
			                            input_names.size(), model.n_features);
		}
		for (idx_t i = 0; i < model.n_features; i++) {
			indices.push_back(i);
		}
		return indices;
	}

	for (const auto &feature : model.feature_names) {
		bool found = false;
		for (idx_t i = 0; i < input_names.size(); i++) {
			if (StringUtil::CIEquals(feature, input_names[i])) {
				indices.push_back(i);
				found = true;
				break;
			}
		}
		if (!found) {
			throw InvalidInputException("ml_predict input table is missing feature column '%s'", feature);
		}
	}
	return indices;
}

static vector<idx_t> ResolveFeatureIndices(const BoostedTreeRegressorModel &model, const vector<string> &input_names) {
	vector<idx_t> indices;
	indices.reserve(model.feature_names.size());
	for (const auto &feature : model.feature_names) {
		bool found = false;
		for (idx_t i = 0; i < input_names.size(); i++) {
			if (StringUtil::CIEquals(feature, input_names[i])) {
				indices.push_back(i);
				found = true;
				break;
			}
		}
		if (!found) {
			throw InvalidInputException("ml_predict input table is missing feature column '%s'", feature);
		}
	}
	return indices;
}

static unique_ptr<FunctionData> MlPredictBind(ClientContext &, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("ml_predict requires a non-NULL model blob");
	}
	auto model_blob = StringValue::Get(input.inputs[0]);
	if (input.input_table_types.empty()) {
		throw InvalidInputException("ml_predict requires a TABLE input");
	}

	if (IsBoostedTreeRegressorModelBlob(model_blob)) {
		auto model = DeserializeBoostedTreeRegressorModel(model_blob);
		auto feature_indices = ResolveFeatureIndices(model, input.input_table_names);
		names.push_back(model.label);
		return_types.push_back(LogicalType::DOUBLE);
		return make_uniq<MlPredictBindData>("boosted_tree_regressor", PcaModel(), std::move(model),
		                                   std::move(feature_indices));
	}

	auto model = DeserializePcaModel(model_blob);
	if (model.model_type != "pca") {
		throw NotImplementedException("ml_predict is not supported for model_type '%s'", model.model_type);
	}
	auto feature_indices = ResolveFeatureIndices(model, input.input_table_names);
	for (idx_t i = 0; i < model.n_components; i++) {
		names.push_back("principal_component_" + to_string(i + 1));
		return_types.push_back(LogicalType::DOUBLE);
	}
	return make_uniq<MlPredictBindData>("pca", std::move(model), BoostedTreeRegressorModel(),
	                                   std::move(feature_indices));
}

static unique_ptr<GlobalTableFunctionState> MlPredictInitGlobal(ClientContext &, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<MlPredictBindData>();
	return make_uniq<MlPredictGlobalState>(bind_data);
}

static OperatorResultType MlPredictFunction(ExecutionContext &, TableFunctionInput &data, DataChunk &input,
                                            DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<MlPredictBindData>();
	auto &global_state = data.global_state->Cast<MlPredictGlobalState>();
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}
	if (bind_data.model_type == "boosted_tree_regressor") {
		vector<float> features;
		auto n_features = bind_data.feature_indices.size();
		features.reserve(input.size() * n_features);
		for (idx_t row = 0; row < input.size(); row++) {
			for (idx_t col = 0; col < n_features; col++) {
				auto value = input.GetValue(bind_data.feature_indices[col], row);
				if (value.IsNull()) {
					throw InvalidInputException("ml_predict does not support NULL feature values");
				}
				features.push_back(
				    static_cast<float>(DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE))));
			}
		}

		auto predictions = global_state.predictor->Predict(features, input.size(), n_features);
		output.SetCardinality(input.size());
		auto out_ptr = FlatVector::GetData<double>(output.data[0]);
		for (idx_t row = 0; row < input.size(); row++) {
			out_ptr[row] = predictions[row];
		}
		return OperatorResultType::NEED_MORE_INPUT;
	}

	Eigen::MatrixXd features(input.size(), bind_data.pca_model.n_features);
	for (idx_t row = 0; row < input.size(); row++) {
		for (idx_t col = 0; col < bind_data.feature_indices.size(); col++) {
			auto value = input.GetValue(bind_data.feature_indices[col], row);
			if (value.IsNull()) {
				throw InvalidInputException("ml_predict does not support NULL feature values");
			}
			features(UnsafeNumericCast<int64_t>(row), UnsafeNumericCast<int64_t>(col)) =
			    DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		}
	}

	auto transformed = Transform(bind_data.pca_model, features);
	output.SetCardinality(input.size());
	for (idx_t col = 0; col < bind_data.pca_model.n_components; col++) {
		auto out_ptr = FlatVector::GetData<double>(output.data[col]);
		for (idx_t row = 0; row < input.size(); row++) {
			out_ptr[row] = transformed(UnsafeNumericCast<int64_t>(row), UnsafeNumericCast<int64_t>(col));
		}
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

} // namespace

void RegisterMlPredict(ExtensionLoader &loader) {
	vector<LogicalType> predict_args = {LogicalType(LogicalTypeId::BLOB), LogicalType(LogicalTypeId::TABLE)};
	TableFunction predict_fn("ml_predict", predict_args, nullptr, MlPredictBind, MlPredictInitGlobal, nullptr);
	predict_fn.in_out_function = MlPredictFunction;
	loader.RegisterFunction(std::move(predict_fn));
}

} // namespace ml
} // namespace duckdb
