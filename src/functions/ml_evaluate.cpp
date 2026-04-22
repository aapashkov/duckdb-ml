// Copyright 2026
//
// ml_evaluate implementations for stored and recomputed PCA metrics.
//
// Metric: total_explained_variance_ratio.
// Library: Eigen.
// Constraints: streaming/batched recomputation with TABLE inputs.

#include "ml/register.hpp"

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
	MlEvaluateBindData(PcaModel model_p, vector<idx_t> feature_indices_p)
	    : model(std::move(model_p)), feature_indices(std::move(feature_indices_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MlEvaluateBindData>(model, feature_indices);
	}

	bool Equals(const FunctionData &other) const override {
		auto &rhs = other.Cast<MlEvaluateBindData>();
		return model.n_features == rhs.model.n_features && model.n_components == rhs.model.n_components &&
		       feature_indices == rhs.feature_indices;
	}

	PcaModel model;
	vector<idx_t> feature_indices;
};

struct MlEvaluateGlobalState : public GlobalTableFunctionState {
	explicit MlEvaluateGlobalState(const PcaModel &model)
	    : original_moments(model.n_features), projected_moments(model.n_components) {
	}

	mutex lock;
	RunningMoments original_moments;
	RunningMoments projected_moments;
	bool emitted = false;
};

static vector<idx_t> ResolveFeatureIndices(const PcaModel &model, const vector<string> &input_names) {
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

static unique_ptr<FunctionData> MlEvaluateBind(ClientContext &, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("ml_evaluate requires a non-NULL model blob");
	}
	auto model = DeserializePcaModel(StringValue::Get(input.inputs[0]));
	if (model.model_type != "pca") {
		throw NotImplementedException("ml_evaluate only supports model_type 'pca'");
	}
	auto feature_indices = ResolveFeatureIndices(model, input.input_table_names);
	names.emplace_back("total_explained_variance_ratio");
	return_types.emplace_back(LogicalType::DOUBLE);
	return make_uniq<MlEvaluateBindData>(std::move(model), std::move(feature_indices));
}

static unique_ptr<GlobalTableFunctionState> MlEvaluateInitGlobal(ClientContext &, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<MlEvaluateBindData>();
	return make_uniq<MlEvaluateGlobalState>(bind_data.model);
}

static OperatorResultType MlEvaluateFunction(ExecutionContext &, TableFunctionInput &input_data, DataChunk &input,
                                             DataChunk &) {
	auto &bind_data = input_data.bind_data->Cast<MlEvaluateBindData>();
	auto &global_state = input_data.global_state->Cast<MlEvaluateGlobalState>();
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	Eigen::MatrixXd features(input.size(), bind_data.model.n_features);
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
	Eigen::MatrixXd projected = features * bind_data.model.components.transpose();

	lock_guard<mutex> guard(global_state.lock);
	global_state.original_moments.Update(features);
	global_state.projected_moments.Update(projected);
	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType MlEvaluateFinalize(ExecutionContext &, TableFunctionInput &input_data,
                                                     DataChunk &output) {
	auto &global_state = input_data.global_state->Cast<MlEvaluateGlobalState>();
	lock_guard<mutex> guard(global_state.lock);
	if (global_state.emitted) {
		return OperatorFinalizeResultType::FINISHED;
	}
	double total_var = global_state.original_moments.TotalSampleVariance();
	double explained_var = global_state.projected_moments.TotalSampleVariance();
	double ratio = total_var > 0 ? explained_var / total_var : 0.0;

	output.SetCardinality(1);
	FlatVector::GetData<double>(output.data[0])[0] = ratio;
	global_state.emitted = true;
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

static void MlEvaluateScalarFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto out = FlatVector::GetData<double>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		auto value = args.data[0].GetValue(i);
		if (value.IsNull()) {
			FlatVector::Validity(result).SetInvalid(i);
			continue;
		}
		auto model = DeserializePcaModel(StringValue::Get(value));
		out[i] = model.training_total_explained_variance_ratio;
	}
}

} // namespace

void RegisterMlEvaluate(ExtensionLoader &loader) {
	ScalarFunction evaluate_scalar("ml_evaluate", {LogicalType::BLOB}, LogicalType::DOUBLE, MlEvaluateScalarFunction,
	                               nullptr, nullptr, nullptr, nullptr, LogicalType(LogicalTypeId::INVALID),
	                               FunctionStability::CONSISTENT, FunctionNullHandling::SPECIAL_HANDLING);
	loader.RegisterFunction(std::move(evaluate_scalar));

	TableFunction evaluate_table("ml_evaluate", {LogicalType::BLOB, LogicalType::TABLE}, nullptr, MlEvaluateBind,
	                             MlEvaluateInitGlobal, nullptr);
	evaluate_table.in_out_function = MlEvaluateFunction;
	evaluate_table.in_out_function_final = MlEvaluateFinalize;
	loader.RegisterFunction(std::move(evaluate_table));
}

} // namespace ml
} // namespace duckdb
