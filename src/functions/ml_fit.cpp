// Copyright 2026
//
// ml_fit table function for Incremental PCA training.
//
// Algorithm: sklearn-style IncrementalPCA partial_fit over DataChunk batches.
// Library: Eigen-backed implementation in src/algorithms/pca.cpp.
// Constraints: no full-table materialization.

#include "ml/register.hpp"

#include "ml/pca.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <mutex>

namespace duckdb {
namespace ml {
namespace {

struct MlFitBindData : public FunctionData {
	MlFitBindData(FitOptions options_p, vector<string> feature_names_p)
	    : options(std::move(options_p)), feature_names(std::move(feature_names_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MlFitBindData>(options, feature_names);
	}

	bool Equals(const FunctionData &other) const override {
		auto &rhs = other.Cast<MlFitBindData>();
		return options.model_type == rhs.options.model_type && options.num_components == rhs.options.num_components &&
		       options.whiten == rhs.options.whiten && feature_names == rhs.feature_names;
	}

	FitOptions options;
	vector<string> feature_names;
};

struct MlFitGlobalState : public GlobalTableFunctionState {
	explicit MlFitGlobalState(const MlFitBindData &bind)
	    : trainer(bind.options.num_components, bind.options.whiten, bind.feature_names.size()),
	      training_moments(bind.feature_names.size()) {
	}

	mutex lock;
	IncrementalPca trainer;
	RunningMoments training_moments;
	vector<double> pending_data;
	idx_t pending_rows = 0;
	bool emitted = false;
};

static void FlushPendingRows(MlFitGlobalState &state) {
	if (state.pending_rows == 0) {
		return;
	}
	auto n_features = state.trainer.NFeatures();
	Eigen::MatrixXd batch(state.pending_rows, n_features);
	for (idx_t row = 0; row < state.pending_rows; row++) {
		for (idx_t col = 0; col < n_features; col++) {
			batch(UnsafeNumericCast<int64_t>(row), UnsafeNumericCast<int64_t>(col)) =
			    state.pending_data[row * n_features + col];
		}
	}
	state.training_moments.Update(batch);
	state.trainer.PartialFit(batch);
	state.pending_data.clear();
	state.pending_rows = 0;
}

static unique_ptr<FunctionData> MlFitBind(ClientContext &, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("ml_fit options cannot be NULL");
	}
	auto options = ParseFitOptions(input.inputs[0]);

	if (input.input_table_types.empty()) {
		throw InvalidInputException("ml_fit requires a TABLE argument with at least one feature column");
	}

	vector<string> feature_names = input.input_table_names;
	if (feature_names.empty()) {
		feature_names.reserve(input.input_table_types.size());
		for (idx_t i = 0; i < input.input_table_types.size(); i++) {
			feature_names.push_back("feature_" + to_string(i + 1));
		}
	}

	names.emplace_back("model");
	return_types.emplace_back(LogicalType::BLOB);
	return make_uniq<MlFitBindData>(std::move(options), std::move(feature_names));
}

static unique_ptr<GlobalTableFunctionState> MlFitInitGlobal(ClientContext &, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<MlFitBindData>();
	return make_uniq<MlFitGlobalState>(bind_data);
}

static OperatorResultType MlFitFunction(ExecutionContext &, TableFunctionInput &input_data, DataChunk &input,
                                        DataChunk &) {
	auto &global_state = input_data.global_state->Cast<MlFitGlobalState>();
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	static constexpr idx_t FLUSH_ROWS = 512;
	lock_guard<mutex> guard(global_state.lock);
	global_state.pending_data.reserve(global_state.pending_data.size() + input.size() * input.ColumnCount());
	for (idx_t row = 0; row < input.size(); row++) {
		for (idx_t col = 0; col < input.ColumnCount(); col++) {
			auto value = input.GetValue(col, row);
			if (value.IsNull()) {
				throw InvalidInputException("ml_fit does not support NULL feature values");
			}
			global_state.pending_data.push_back(DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE)));
		}
		global_state.pending_rows++;
	}
	if (global_state.pending_rows >= FLUSH_ROWS) {
		FlushPendingRows(global_state);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType MlFitFinalize(ExecutionContext &, TableFunctionInput &input_data, DataChunk &output) {
	auto &bind_data = input_data.bind_data->Cast<MlFitBindData>();
	auto &global_state = input_data.global_state->Cast<MlFitGlobalState>();
	lock_guard<mutex> guard(global_state.lock);
	if (global_state.emitted) {
		return OperatorFinalizeResultType::FINISHED;
	}
	FlushPendingRows(global_state);

	output.SetCardinality(1);
	if (!global_state.trainer.IsInitialized()) {
		FlatVector::Validity(output.data[0]).SetInvalid(0);
	} else {
		auto model = global_state.trainer.ToModel(bind_data.feature_names);
		auto total_var = global_state.training_moments.TotalSampleVariance();
		if (total_var > 0) {
			model.training_total_explained_variance_ratio = model.explained_variance.sum() / total_var;
		} else {
			model.training_total_explained_variance_ratio = 0.0;
		}
		auto blob = SerializePcaModel(model);
		FlatVector::GetData<string_t>(output.data[0])[0] =
		    StringVector::AddStringOrBlob(output.data[0], blob.data(), blob.size());
	}
	global_state.emitted = true;
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

} // namespace

void RegisterMlFit(ExtensionLoader &loader) {
	TableFunction fit_fn("ml_fit", {LogicalType::ANY, LogicalType::ANY, LogicalType::TABLE}, nullptr, MlFitBind,
	                     MlFitInitGlobal, nullptr);
	fit_fn.in_out_function = MlFitFunction;
	fit_fn.in_out_function_final = MlFitFinalize;
	loader.RegisterFunction(std::move(fit_fn));
}

} // namespace ml
} // namespace duckdb
