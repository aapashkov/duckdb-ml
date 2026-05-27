// Copyright 2026
//
// ml_explain placeholder for PCA.
//
// PCA does not provide per-row feature attribution in this extension,
// so this function is registered and fails with a user-facing error.

#include "ml/register.hpp"

#include "ml/boosted_tree_regressor.hpp"
#include "ml/model_registry.hpp"
#include "ml/pca.hpp"

#include "duckdb.hpp"

namespace duckdb {
namespace ml {
namespace {

struct MlExplainBindData : public FunctionData {
	MlExplainBindData(string model_type_p, PcaModel pca_model_p, BoostedTreeRegressorModel boosted_model_p,
	                 vector<idx_t> feature_indices_p)
	    : model_type(std::move(model_type_p)), pca_model(std::move(pca_model_p)),
	      boosted_model(std::move(boosted_model_p)), feature_indices(std::move(feature_indices_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MlExplainBindData>(model_type, pca_model, boosted_model, feature_indices);
	}

	bool Equals(const FunctionData &other) const override {
		auto &rhs = other.Cast<MlExplainBindData>();
		return model_type == rhs.model_type && feature_indices == rhs.feature_indices;
	}

	string model_type;
	PcaModel pca_model;
	BoostedTreeRegressorModel boosted_model;
	vector<idx_t> feature_indices;
};

struct MlExplainGlobalState : public GlobalTableFunctionState {
	explicit MlExplainGlobalState(const MlExplainBindData &bind_data) {
		if (bind_data.model_type == "boosted_tree_regressor") {
			predictor = make_uniq<BoostedTreePredictor>(bind_data.boosted_model);
		}
	}

	unique_ptr<BoostedTreePredictor> predictor;
};

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
			throw InvalidInputException("ml_explain input table is missing feature column '%s'", feature);
		}
	}
	return indices;
}

static string ReadModelName(const Value &value) {
	if (value.IsNull()) {
		throw InvalidInputException("ml_explain requires a non-NULL model name");
	}
	auto model_name = StringValue::Get(value);
	StringUtil::Trim(model_name);
	if (model_name.empty()) {
		throw InvalidInputException("ml_explain requires a non-empty model name");
	}
	return model_name;
}

static idx_t ReadRequestedVersion(TableFunctionBindInput &input) {
	auto entry = input.named_parameters.find("version");
	if (entry == input.named_parameters.end()) {
		return 0;
	}
	if (entry->second.IsNull()) {
		throw InvalidInputException("ml_explain version must be a non-NULL integer");
	}
	auto raw_version = BigIntValue::Get(entry->second.DefaultCastAs(LogicalType::BIGINT));
	if (raw_version < 0) {
		throw InvalidInputException("ml_explain version must be >= 0");
	}
	return UnsafeNumericCast<idx_t>(raw_version);
}

static unique_ptr<FunctionData> MlExplainBind(ClientContext &, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty()) {
		throw InvalidInputException("ml_explain requires a model name argument");
	}
	auto model_name = ReadModelName(input.inputs[0]);
	auto requested_version = ReadRequestedVersion(input);
	auto lookup = LoadModelRegistryEntry(model_name, requested_version);
	auto blob = lookup.blob;
	if (IsBoostedTreeRegressorModelBlob(blob)) {
		auto model = DeserializeBoostedTreeRegressorModel(blob);
		auto feature_indices = ResolveFeatureIndices(model, input.input_table_names);
		names.emplace_back("baseline_prediction_value");
		return_types.emplace_back(LogicalType::DOUBLE);
		for (const auto &feature : model.feature_names) {
			names.push_back(feature + "_attribution");
			return_types.emplace_back(LogicalType::DOUBLE);
		}
		return make_uniq<MlExplainBindData>("boosted_tree_regressor", PcaModel(), std::move(model),
		                                   std::move(feature_indices));
	}

	auto model = DeserializePcaModel(blob);
	if (model.model_type != "pca") {
		throw NotImplementedException("ml_explain is not supported for model_type '%s'", model.model_type);
	}

	names.emplace_back("feature_attribution");
	return_types.emplace_back(LogicalType::DOUBLE);
	return make_uniq<MlExplainBindData>("pca", std::move(model), BoostedTreeRegressorModel(), vector<idx_t>());
}

static unique_ptr<GlobalTableFunctionState> MlExplainInitGlobal(ClientContext &, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<MlExplainBindData>();
	return make_uniq<MlExplainGlobalState>(bind_data);
}

static OperatorResultType MlExplainFunction(ExecutionContext &, TableFunctionInput &input_data, DataChunk &input,
                                            DataChunk &output) {
	auto &bind_data = input_data.bind_data->Cast<MlExplainBindData>();
	auto &global_state = input_data.global_state->Cast<MlExplainGlobalState>();

	if (bind_data.model_type == "pca") {
		throw InvalidInputException(
		    "ml_explain is not supported for model_type 'pca': feature attribution is undefined for PCA");
	}
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto n_features = bind_data.feature_indices.size();
	vector<float> features;
	features.reserve(input.size() * n_features);
	for (idx_t row = 0; row < input.size(); row++) {
		for (idx_t col = 0; col < n_features; col++) {
			auto value = input.GetValue(bind_data.feature_indices[col], row);
			if (value.IsNull()) {
				throw InvalidInputException("ml_explain does not support NULL feature values");
			}
			features.push_back(static_cast<float>(DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE))));
		}
	}

	auto contributions = global_state.predictor->PredictContributions(features, input.size(), n_features);
	output.SetCardinality(input.size());
	for (idx_t row = 0; row < input.size(); row++) {
		auto row_offset = row * (n_features + 1);
		FlatVector::GetData<double>(output.data[0])[row] = contributions[row_offset + n_features];
		for (idx_t col = 0; col < n_features; col++) {
			FlatVector::GetData<double>(output.data[col + 1])[row] = contributions[row_offset + col];
		}
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

} // namespace

void RegisterMlExplain(ExtensionLoader &loader) {
	vector<LogicalType> explain_args = {LogicalType(LogicalTypeId::VARCHAR), LogicalType(LogicalTypeId::TABLE)};
	TableFunction explain("ml_explain", explain_args, nullptr, MlExplainBind, MlExplainInitGlobal, nullptr);
	explain.named_parameters["version"] = LogicalType::BIGINT;
	explain.in_out_function = MlExplainFunction;
	loader.RegisterFunction(std::move(explain));
}

} // namespace ml
} // namespace duckdb
