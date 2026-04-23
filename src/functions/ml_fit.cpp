// Copyright 2026
//
// ml_fit table function for Incremental PCA training.
//
// Algorithm: sklearn-style IncrementalPCA partial_fit over DataChunk batches.
// Library: Eigen-backed implementation in src/algorithms/pca.cpp.
// Constraints: no full-table materialization.

#include "ml/register.hpp"

#include "ml/boosted_tree_regressor.hpp"
#include "ml/pca.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"

#include <chrono>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>

namespace duckdb {
namespace ml {
namespace {

static bool IsNumericType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return true;
	default:
		return false;
	}
}

static string CreateBoostedTreeCacheDir() {
	char dir_template[] = "/tmp/duckdb_ml_boosted_tree_XXXXXX";
	auto *dir_path = mkdtemp(dir_template);
	if (!dir_path) {
		throw IOException("Could not create temporary directory for boosted_tree_regressor training");
	}
	string cache_dir(dir_path);
	string xgb_cache_dir = cache_dir + "/xgb_cache";
	if (mkdir(xgb_cache_dir.c_str(), 0700) != 0) {
		throw IOException("Could not create XGBoost cache directory '%s'", xgb_cache_dir);
	}
	return cache_dir;
}

struct MlFitBindData : public FunctionData {
	MlFitBindData(string model_type_p, FitOptions pca_options_p, BoostedTreeFitOptions boosted_options_p,
	              vector<idx_t> feature_indices_p, idx_t label_index_p, vector<string> model_feature_names_p)
	    : model_type(std::move(model_type_p)), pca_options(std::move(pca_options_p)),
	      boosted_options(std::move(boosted_options_p)), feature_indices(std::move(feature_indices_p)),
	      label_index(label_index_p), model_feature_names(std::move(model_feature_names_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MlFitBindData>(model_type, pca_options, boosted_options, feature_indices, label_index,
		                               model_feature_names);
	}

	bool Equals(const FunctionData &other) const override {
		auto &rhs = other.Cast<MlFitBindData>();
		return model_type == rhs.model_type && feature_indices == rhs.feature_indices && label_index == rhs.label_index &&
		       model_feature_names == rhs.model_feature_names;
	}

	string model_type;
	FitOptions pca_options;
	BoostedTreeFitOptions boosted_options;
	vector<idx_t> feature_indices;
	idx_t label_index;
	vector<string> model_feature_names;
};

struct MlFitGlobalState : public GlobalTableFunctionState {
	explicit MlFitGlobalState(const MlFitBindData &bind)
	    : model_type(bind.model_type), label_index(bind.label_index), feature_indices(bind.feature_indices),
	      model_feature_names(bind.model_feature_names) {
		if (model_type == "pca") {
			trainer = make_uniq<IncrementalPca>(bind.pca_options.num_components, bind.pca_options.whiten,
			                                   bind.model_feature_names.size());
		} else {
			cache_dir = CreateBoostedTreeCacheDir();
		}
	}

	~MlFitGlobalState() override {
		for (const auto &batch_file : batch_files) {
			unlink(batch_file.c_str());
		}
	}

	mutex lock;
	string model_type;
	idx_t label_index;
	vector<idx_t> feature_indices;
	vector<string> model_feature_names;

	unique_ptr<IncrementalPca> trainer;
	vector<double> pending_data;

	string cache_dir;
	vector<string> batch_files;
	vector<float> pending_feature_data;
	vector<float> pending_labels;
	idx_t next_batch_id = 0;

	idx_t pending_rows = 0;
	bool emitted = false;
};

static void FlushPendingRows(MlFitGlobalState &state) {
	if (state.pending_rows == 0) {
		return;
	}
	auto n_features = state.trainer->NFeatures();
	Eigen::MatrixXd batch(state.pending_rows, n_features);
	for (idx_t row = 0; row < state.pending_rows; row++) {
		for (idx_t col = 0; col < n_features; col++) {
			batch(UnsafeNumericCast<int64_t>(row), UnsafeNumericCast<int64_t>(col)) =
			    state.pending_data[row * n_features + col];
		}
	}
	state.trainer->PartialFit(batch);
	state.pending_data.clear();
	state.pending_rows = 0;
}

static void FlushPendingBoostedTreeRows(MlFitGlobalState &state) {
	if (state.pending_rows == 0) {
		return;
	}
	auto batch_path = state.cache_dir + "/batch_" + to_string(state.next_batch_id++) + ".bin";
	WriteBoostedTreeBatchFile(batch_path, state.pending_rows, state.feature_indices.size(), state.pending_feature_data,
	                         state.pending_labels);
	state.batch_files.push_back(batch_path);
	state.pending_feature_data.clear();
	state.pending_labels.clear();
	state.pending_rows = 0;
}

static unique_ptr<FunctionData> MlFitBind(ClientContext &, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	auto model_type = ParseModelType(input.inputs[0]);

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

	FitOptions pca_options;
	BoostedTreeFitOptions boosted_options;
	vector<idx_t> feature_indices;
	idx_t label_index = DConstants::INVALID_INDEX;
	vector<string> model_feature_names;

	if (model_type == "pca") {
		pca_options = ParseFitOptions(input.inputs[0]);
		model_feature_names = feature_names;
		for (idx_t i = 0; i < feature_names.size(); i++) {
			feature_indices.push_back(i);
		}
	} else if (model_type == "boosted_tree_regressor") {
		boosted_options = ParseBoostedTreeFitOptions(input.inputs[0]);
		for (idx_t i = 0; i < input.input_table_types.size(); i++) {
			if (!IsNumericType(input.input_table_types[i])) {
				throw InvalidInputException("ml_fit for boosted_tree_regressor only supports numeric columns. "
				                            "Column '%s' has type %s",
				                            feature_names[i], input.input_table_types[i].ToString());
			}
		}
		for (idx_t i = 0; i < feature_names.size(); i++) {
			if (StringUtil::CIEquals(feature_names[i], boosted_options.label)) {
				label_index = i;
				break;
			}
		}
		if (label_index == DConstants::INVALID_INDEX) {
			throw InvalidInputException("options.label '%s' does not exist in the training input", boosted_options.label);
		}
		for (idx_t i = 0; i < feature_names.size(); i++) {
			if (i != label_index) {
				feature_indices.push_back(i);
				model_feature_names.push_back(feature_names[i]);
			}
		}
		if (feature_indices.empty()) {
			throw InvalidInputException("boosted_tree_regressor requires at least one feature column besides label");
		}
	} else {
		throw NotImplementedException("Unsupported model_type '%s'", model_type);
	}

	names.emplace_back("model");
	return_types.emplace_back(LogicalType::BLOB);
	return make_uniq<MlFitBindData>(std::move(model_type), std::move(pca_options), std::move(boosted_options),
	                               std::move(feature_indices), label_index, std::move(model_feature_names));
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
	if (global_state.model_type == "pca") {
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
	} else {
		auto n_features = global_state.feature_indices.size();
		global_state.pending_feature_data.reserve(global_state.pending_feature_data.size() + input.size() * n_features);
		global_state.pending_labels.reserve(global_state.pending_labels.size() + input.size());
		for (idx_t row = 0; row < input.size(); row++) {
			auto label_value = input.GetValue(global_state.label_index, row);
			if (label_value.IsNull()) {
				throw InvalidInputException("ml_fit for boosted_tree_regressor does not support NULL label values");
			}
			global_state.pending_labels.push_back(
			    static_cast<float>(DoubleValue::Get(label_value.DefaultCastAs(LogicalType::DOUBLE))));
			for (idx_t col = 0; col < global_state.feature_indices.size(); col++) {
				auto value = input.GetValue(global_state.feature_indices[col], row);
				if (value.IsNull()) {
					throw InvalidInputException("ml_fit for boosted_tree_regressor does not support NULL feature values");
				}
				global_state.pending_feature_data.push_back(
				    static_cast<float>(DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE))));
			}
			global_state.pending_rows++;
		}
		if (global_state.pending_rows >= FLUSH_ROWS) {
			FlushPendingBoostedTreeRows(global_state);
		}
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

	output.SetCardinality(1);
	if (global_state.model_type == "pca") {
		FlushPendingRows(global_state);
		if (!global_state.trainer->IsInitialized()) {
			FlatVector::Validity(output.data[0]).SetInvalid(0);
		} else {
			auto model = global_state.trainer->ToModel(bind_data.model_feature_names);
			auto blob = SerializePcaModel(model);
			FlatVector::GetData<string_t>(output.data[0])[0] =
			    StringVector::AddStringOrBlob(output.data[0], blob.data(), blob.size());
		}
	} else {
		FlushPendingBoostedTreeRows(global_state);
		if (global_state.batch_files.empty()) {
			FlatVector::Validity(output.data[0]).SetInvalid(0);
		} else {
			auto model = TrainBoostedTreeRegressorFromBatchFiles(bind_data.boosted_options,
			                                                    bind_data.model_feature_names,
			                                                    global_state.batch_files,
			                                                    global_state.cache_dir + "/xgb_cache/cache");
			auto blob = SerializeBoostedTreeRegressorModel(model);
			FlatVector::GetData<string_t>(output.data[0])[0] =
			    StringVector::AddStringOrBlob(output.data[0], blob.data(), blob.size());
		}
	}

	global_state.emitted = true;
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

} // namespace

void RegisterMlFit(ExtensionLoader &loader) {
	vector<LogicalType> fit_args = {LogicalType(LogicalTypeId::ANY), LogicalType(LogicalTypeId::ANY),
	                                LogicalType(LogicalTypeId::TABLE)};
	TableFunction fit_fn("ml_fit", fit_args, nullptr, MlFitBind, MlFitInitGlobal, nullptr);
	fit_fn.in_out_function = MlFitFunction;
	fit_fn.in_out_function_final = MlFitFinalize;
	loader.RegisterFunction(std::move(fit_fn));
}

} // namespace ml
} // namespace duckdb
