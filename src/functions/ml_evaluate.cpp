// Copyright 2026
//
// ml_evaluate implementations for PCA and boosted tree regressor.
//
// Metrics:
// - PCA: total_explained_variance_ratio
// - boosted_tree_regressor: MAE, MSE, MSLE, median absolute error, R2, explained variance
//
// Constraints: streaming/batched recomputation with TABLE inputs.

#include "ml/register.hpp"

#include "ml/boosted_tree_regressor.hpp"
#include "ml/pca.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"

#include <mutex>

namespace duckdb {
namespace ml {
namespace {

struct MlEvaluateBindData : public FunctionData {
	MlEvaluateBindData(string model_type_p, bool has_input_table_p, PcaModel pca_model_p,
	                   BoostedTreeRegressorModel boosted_model_p, vector<idx_t> feature_indices_p,
	                   idx_t label_index_p)
	    : model_type(std::move(model_type_p)), has_input_table(has_input_table_p), pca_model(std::move(pca_model_p)),
	      boosted_model(std::move(boosted_model_p)), feature_indices(std::move(feature_indices_p)),
	      label_index(label_index_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MlEvaluateBindData>(model_type, has_input_table, pca_model, boosted_model, feature_indices,
		                                    label_index);
	}

	bool Equals(const FunctionData &other) const override {
		auto &rhs = other.Cast<MlEvaluateBindData>();
		return model_type == rhs.model_type && has_input_table == rhs.has_input_table &&
		       feature_indices == rhs.feature_indices && label_index == rhs.label_index;
	}

	string model_type;
	bool has_input_table;
	PcaModel pca_model;
	BoostedTreeRegressorModel boosted_model;
	vector<idx_t> feature_indices;
	idx_t label_index;
};

struct MlEvaluateGlobalState : public GlobalTableFunctionState {
	explicit MlEvaluateGlobalState(const MlEvaluateBindData &bind_data) {
		if (!bind_data.has_input_table) {
			return;
		}
		if (bind_data.model_type == "pca") {
			original_moments = make_uniq<RunningMoments>(bind_data.pca_model.n_features);
			projected_moments = make_uniq<RunningMoments>(bind_data.pca_model.n_components);
		} else {
			predictor = make_uniq<BoostedTreePredictor>(bind_data.boosted_model);
			metrics_acc = make_uniq<RegressionMetricsAccumulator>();
		}
	}

	mutex lock;
	unique_ptr<RunningMoments> original_moments;
	unique_ptr<RunningMoments> projected_moments;
	unique_ptr<BoostedTreePredictor> predictor;
	unique_ptr<RegressionMetricsAccumulator> metrics_acc;
	bool emitted = false;
};

static void AddRegressionMetricColumns(vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("mean_absolute_error");
	names.emplace_back("mean_squared_error");
	names.emplace_back("mean_squared_log_error");
	names.emplace_back("median_absolute_error");
	names.emplace_back("r2_score");
	names.emplace_back("explained_variance");
	for (idx_t i = 0; i < 6; i++) {
		return_types.emplace_back(LogicalType::DOUBLE);
	}
}

static void WriteRegressionMetricRow(DataChunk &output, const RegressionMetrics &metrics) {
	output.SetCardinality(1);
	FlatVector::GetData<double>(output.data[0])[0] = metrics.mean_absolute_error;
	FlatVector::GetData<double>(output.data[1])[0] = metrics.mean_squared_error;
	FlatVector::GetData<double>(output.data[2])[0] = metrics.mean_squared_log_error;
	FlatVector::GetData<double>(output.data[3])[0] = metrics.median_absolute_error;
	FlatVector::GetData<double>(output.data[4])[0] = metrics.r2_score;
	FlatVector::GetData<double>(output.data[5])[0] = metrics.explained_variance;
}

static vector<idx_t> ResolvePcaFeatureIndices(const PcaModel &model, const vector<string> &input_names) {
	vector<idx_t> indices;
	indices.reserve(model.n_features);
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
			throw InvalidInputException("ml_evaluate input table is missing feature column '%s'", feature);
		}
	}
	if (indices.empty()) {
		if (input_names.size() < model.n_features) {
			throw InvalidInputException("ml_evaluate input has %llu columns, model expects %llu", input_names.size(),
			                            model.n_features);
		}
		for (idx_t i = 0; i < model.n_features; i++) {
			indices.push_back(i);
		}
	}
	return indices;
}

static vector<idx_t> ResolveBoostedFeatureIndices(const BoostedTreeRegressorModel &model,
                                                  const vector<string> &input_names) {
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
			throw InvalidInputException("ml_evaluate input table is missing feature column '%s'", feature);
		}
	}
	return indices;
}

static idx_t ResolveLabelIndex(const BoostedTreeRegressorModel &model, const vector<string> &input_names) {
	for (idx_t i = 0; i < input_names.size(); i++) {
		if (StringUtil::CIEquals(model.label, input_names[i])) {
			return i;
		}
	}
	throw InvalidInputException("ml_evaluate input table is missing label column '%s'", model.label);
}

static unique_ptr<FunctionData> MlEvaluateBindWithTable(ClientContext &, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("ml_evaluate requires a non-NULL model blob");
	}
	if (input.input_table_types.empty()) {
		throw InvalidInputException("ml_evaluate(model, table) requires a TABLE input");
	}

	auto blob = StringValue::Get(input.inputs[0]);
	if (IsBoostedTreeRegressorModelBlob(blob)) {
		auto model = DeserializeBoostedTreeRegressorModel(blob);
		auto feature_indices = ResolveBoostedFeatureIndices(model, input.input_table_names);
		auto label_index = ResolveLabelIndex(model, input.input_table_names);
		AddRegressionMetricColumns(return_types, names);
		return make_uniq<MlEvaluateBindData>("boosted_tree_regressor", true, PcaModel(), std::move(model),
		                                    std::move(feature_indices), label_index);
	}

	auto model = DeserializePcaModel(blob);
	if (model.model_type != "pca") {
		throw NotImplementedException("ml_evaluate is not supported for model_type '%s'", model.model_type);
	}
	auto feature_indices = ResolvePcaFeatureIndices(model, input.input_table_names);
	names.emplace_back("total_explained_variance_ratio");
	return_types.emplace_back(LogicalType::DOUBLE);
	return make_uniq<MlEvaluateBindData>("pca", true, std::move(model), BoostedTreeRegressorModel(),
	                                    std::move(feature_indices), DConstants::INVALID_INDEX);
}

static unique_ptr<GlobalTableFunctionState> MlEvaluateInitGlobal(ClientContext &, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<MlEvaluateBindData>();
	return make_uniq<MlEvaluateGlobalState>(bind_data);
}

static OperatorResultType MlEvaluateFunction(ExecutionContext &, TableFunctionInput &input_data, DataChunk &input,
                                             DataChunk &) {
	auto &bind_data = input_data.bind_data->Cast<MlEvaluateBindData>();
	auto &global_state = input_data.global_state->Cast<MlEvaluateGlobalState>();
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	if (bind_data.model_type == "pca") {
		Eigen::MatrixXd features(input.size(), bind_data.pca_model.n_features);
		for (idx_t row = 0; row < input.size(); row++) {
			for (idx_t col = 0; col < bind_data.feature_indices.size(); col++) {
				auto value = input.GetValue(bind_data.feature_indices[col], row);
				if (value.IsNull()) {
					throw InvalidInputException("ml_evaluate does not support NULL feature values");
				}
				features(UnsafeNumericCast<int64_t>(row), UnsafeNumericCast<int64_t>(col)) =
				    DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
			}
		}
		Eigen::MatrixXd projected = features * bind_data.pca_model.components.transpose();

		lock_guard<mutex> guard(global_state.lock);
		global_state.original_moments->Update(features);
		global_state.projected_moments->Update(projected);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	vector<float> features;
	vector<float> labels;
	auto n_features = bind_data.feature_indices.size();
	features.reserve(input.size() * n_features);
	labels.reserve(input.size());
	for (idx_t row = 0; row < input.size(); row++) {
		auto label_value = input.GetValue(bind_data.label_index, row);
		if (label_value.IsNull()) {
			throw InvalidInputException("ml_evaluate does not support NULL label values");
		}
		labels.push_back(static_cast<float>(DoubleValue::Get(label_value.DefaultCastAs(LogicalType::DOUBLE))));
		for (idx_t col = 0; col < n_features; col++) {
			auto value = input.GetValue(bind_data.feature_indices[col], row);
			if (value.IsNull()) {
				throw InvalidInputException("ml_evaluate does not support NULL feature values");
			}
			features.push_back(static_cast<float>(DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE))));
		}
	}

	auto predictions = global_state.predictor->Predict(features, input.size(), n_features);
	lock_guard<mutex> guard(global_state.lock);
	global_state.metrics_acc->Update(predictions, labels);
	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType MlEvaluateFinalize(ExecutionContext &, TableFunctionInput &input_data,
                                                     DataChunk &output) {
	auto &bind_data = input_data.bind_data->Cast<MlEvaluateBindData>();
	auto &global_state = input_data.global_state->Cast<MlEvaluateGlobalState>();
	lock_guard<mutex> guard(global_state.lock);
	if (global_state.emitted) {
		return OperatorFinalizeResultType::FINISHED;
	}

	if (bind_data.model_type == "pca") {
		double total_var = global_state.original_moments->TotalSampleVariance();
		double explained_var = global_state.projected_moments->TotalSampleVariance();
		double ratio = total_var > 0 ? explained_var / total_var : 0.0;
		output.SetCardinality(1);
		FlatVector::GetData<double>(output.data[0])[0] = ratio;
	} else {
		WriteRegressionMetricRow(output, global_state.metrics_acc->Finalize());
	}

	global_state.emitted = true;
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

static void MlEvaluateScalarFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto &model_vector = args.data[0];
	UnaryExecutor::Execute<string_t, double>(model_vector, result, args.size(), [&](string_t blob_value) {
		auto model_blob = string(blob_value.GetData(), blob_value.GetSize());
		if (IsBoostedTreeRegressorModelBlob(model_blob)) {
			auto model = DeserializeBoostedTreeRegressorModel(model_blob);
			return model.training_metrics.r2_score;
		}
		auto model = DeserializePcaModel(model_blob);
		return model.training_total_explained_variance_ratio;
	});
}

} // namespace

void RegisterMlEvaluate(ExtensionLoader &loader) {
	ScalarFunction evaluate_scalar("ml_evaluate", {LogicalType::BLOB}, LogicalType::DOUBLE, MlEvaluateScalarFunction,
	                               nullptr, nullptr, nullptr, nullptr, LogicalType(LogicalTypeId::INVALID),
	                               FunctionStability::CONSISTENT, FunctionNullHandling::SPECIAL_HANDLING);
	loader.RegisterFunction(std::move(evaluate_scalar));

	vector<LogicalType> evaluate_args = {LogicalType(LogicalTypeId::BLOB), LogicalType(LogicalTypeId::TABLE)};
	TableFunction evaluate_table("ml_evaluate", evaluate_args, nullptr, MlEvaluateBindWithTable, MlEvaluateInitGlobal,
	                             nullptr);
	evaluate_table.in_out_function = MlEvaluateFunction;
	evaluate_table.in_out_function_final = MlEvaluateFinalize;
	loader.RegisterFunction(std::move(evaluate_table));
}

} // namespace ml
} // namespace duckdb
