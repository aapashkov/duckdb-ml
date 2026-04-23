// Copyright 2026
//
// Boosted tree regressor implementation and model serialization.
//
// Algorithm: XGBoost gradient-boosted trees with external-memory DMatrix callbacks.
// Library: XGBoost C API.
// Constraints: no full-table materialization during fit/evaluate/explain.

#include "ml/boosted_tree_regressor.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "xgboost/c_api.h"

namespace duckdb {
namespace ml {
namespace {

static constexpr uint32_t BTR_MAGIC = 0x4d4c4254; // MLBT
static constexpr uint32_t BTR_VERSION = 2;
static constexpr uint32_t BATCH_MAGIC = 0x42545242; // BTRB
static constexpr idx_t MEDIAN_SAMPLE_CAP = 8192;

inline void ThrowXGBoostError(int code, const char *call) {
	if (code != 0) {
		throw InvalidInputException("XGBoost call failed in %s: %s", call, XGBGetLastError());
	}
}

#define XGB_CALL(call) ThrowXGBoostError((call), #call)

struct BlobWriter {
	template <class T>
	void WritePod(const T &value) {
		auto start = blob.size();
		blob.resize(start + sizeof(T));
		memcpy(&blob[start], &value, sizeof(T));
	}

	void WriteString(const string &value) {
		auto len = UnsafeNumericCast<uint64_t>(value.size());
		WritePod(len);
		blob.append(value);
	}

	string blob;
};

struct BlobReader {
	explicit BlobReader(const string &blob_p) : blob(blob_p), offset(0) {
	}

	template <class T>
	T ReadPod() {
		if (offset + sizeof(T) > blob.size()) {
			throw InvalidInputException("Invalid boosted_tree_regressor model blob: truncated payload");
		}
		T value;
		memcpy(&value, blob.data() + offset, sizeof(T));
		offset += sizeof(T);
		return value;
	}

	string ReadString() {
		auto len = ReadPod<uint64_t>();
		if (offset + len > blob.size()) {
			throw InvalidInputException("Invalid boosted_tree_regressor model blob: invalid string length");
		}
		string value(blob.data() + offset, len);
		offset += len;
		return value;
	}

	const string &blob;
	idx_t offset;
};

struct BatchData {
	idx_t n_rows = 0;
	idx_t n_features = 0;
	vector<float> features;
	vector<float> labels;
};

static bool HasOptionKey(const Value &options_value, const string &target_key) {
	auto options_type = options_value.type();
	auto &types = StructType::GetChildTypes(options_type);
	for (idx_t i = 0; i < types.size(); i++) {
		if (StringUtil::Lower(types[i].first) == target_key) {
			return true;
		}
	}
	return false;
}

static string ObjectiveForRegressionObjective(const string &objective) {
	if (objective == "squarederror") {
		return "reg:squarederror";
	}
	if (objective == "squaredlogerror") {
		return "reg:squaredlogerror";
	}
	if (objective == "logistic") {
		return "reg:logistic";
	}
	if (objective == "pseudohubererror") {
		return "reg:pseudohubererror";
	}
	if (objective == "absoluteerror") {
		return "reg:absoluteerror";
	}
	if (objective == "quantileerror") {
		return "reg:quantileerror";
	}
	throw InvalidInputException("options.regression_objective must be one of: squarederror, squaredlogerror, logistic, "
	                            "pseudohubererror, absoluteerror, quantileerror");
}

static void ValidateProbability(const char *name, double value, bool allow_zero) {
	if (allow_zero) {
		if (value < 0.0 || value > 1.0) {
			throw InvalidInputException("options.%s must be in [0.0, 1.0]", name);
		}
	} else {
		if (value <= 0.0 || value > 1.0) {
			throw InvalidInputException("options.%s must be in (0.0, 1.0]", name);
		}
	}
}

static void ValidateNonNegative(const char *name, double value) {
	if (value < 0.0) {
		throw InvalidInputException("options.%s must be >= 0.0", name);
	}
}

static void ValidatePositiveInteger(const char *name, idx_t value) {
	if (value < 1) {
		throw InvalidInputException("options.%s must be >= 1", name);
	}
}

static void ReadBoostedTreeBatchFile(const string &path, BatchData &batch) {
	std::ifstream in(path, std::ios::binary);
	if (!in.good()) {
		throw IOException("Could not read boosted tree training batch file: %s", path);
	}

	uint32_t magic = 0;
	uint64_t rows = 0;
	uint64_t cols = 0;
	in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
	in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
	in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
	if (!in.good() || magic != BATCH_MAGIC) {
		throw InvalidInputException("Invalid boosted tree training batch file format: %s", path);
	}

	batch.n_rows = UnsafeNumericCast<idx_t>(rows);
	batch.n_features = UnsafeNumericCast<idx_t>(cols);
	batch.features.resize(batch.n_rows * batch.n_features);
	batch.labels.resize(batch.n_rows);

	in.read(reinterpret_cast<char *>(batch.features.data()),
	        UnsafeNumericCast<std::streamsize>(batch.features.size() * sizeof(float)));
	in.read(reinterpret_cast<char *>(batch.labels.data()),
	        UnsafeNumericCast<std::streamsize>(batch.labels.size() * sizeof(float)));
	if (!in.good()) {
		throw InvalidInputException("Corrupt boosted tree training batch file: %s", path);
	}
}

static string DenseArrayInterfaceJson(float *data_ptr, idx_t rows, idx_t cols) {
	std::ostringstream ss;
	ss << "{\"data\":[" << reinterpret_cast<uintptr_t>(data_ptr)
	   << ", false],\"shape\":[" << rows << "," << cols
	   << "],\"strides\":null,\"typestr\":\"<f4\",\"version\":3}";
	return ss.str();
}

static string DenseVectorArrayInterfaceJson(float *data_ptr, idx_t len) {
	std::ostringstream ss;
	ss << "{\"data\":[" << reinterpret_cast<uintptr_t>(data_ptr)
	   << ", false],\"shape\":[" << len << "],\"strides\":null,\"typestr\":\"<f4\",\"version\":3}";
	return ss.str();
}

struct ExternalMemoryIterator {
	vector<string> batch_files;
	idx_t n_features;
	idx_t current_batch = 0;
	DMatrixHandle proxy = nullptr;
	BatchData current;
};

static void ExternalMemoryReset(DataIterHandle handle) {
	auto *iter = reinterpret_cast<ExternalMemoryIterator *>(handle);
	iter->current_batch = 0;
}

static int ExternalMemoryNext(DataIterHandle handle) {
	auto *iter = reinterpret_cast<ExternalMemoryIterator *>(handle);
	if (iter->current_batch >= iter->batch_files.size()) {
		iter->current_batch = 0;
		return 0;
	}

	ReadBoostedTreeBatchFile(iter->batch_files[iter->current_batch], iter->current);
	if (iter->current.n_features != iter->n_features) {
		throw InvalidInputException("Boosted tree training batch has inconsistent feature count");
	}

	auto array_interface = DenseArrayInterfaceJson(iter->current.features.data(), iter->current.n_rows, iter->current.n_features);
	auto label_interface = DenseVectorArrayInterfaceJson(iter->current.labels.data(), iter->current.n_rows);
	XGB_CALL(XGProxyDMatrixSetDataDense(iter->proxy, array_interface.c_str()));
	XGB_CALL(XGDMatrixSetInfoFromInterface(iter->proxy, "label", label_interface.c_str()));

	iter->current_batch++;
	return 1;
}

static vector<float> PredictDenseInternal(BoosterHandle booster, const vector<float> &features, idx_t rows, idx_t cols,
                                          idx_t iteration_end, int predict_type) {
	if (rows == 0) {
		return {};
	}
	DMatrixHandle matrix = nullptr;
	XGB_CALL(XGDMatrixCreateFromMat_omp(features.data(), rows, cols, std::numeric_limits<float>::quiet_NaN(), &matrix, 1));

	std::ostringstream config;
	config << "{\"type\":" << predict_type << ",\"training\":false,\"iteration_begin\":0,\"iteration_end\":"
	       << iteration_end << ",\"strict_shape\":false}";

	bst_ulong const *out_shape = nullptr;
	bst_ulong out_dim = 0;
	const float *out_result = nullptr;
	XGB_CALL(XGBoosterPredictFromDMatrix(booster, matrix, config.str().c_str(), &out_shape, &out_dim, &out_result));

	idx_t out_len = 1;
	for (idx_t i = 0; i < out_dim; i++) {
		out_len *= UnsafeNumericCast<idx_t>(out_shape[i]);
	}
	vector<float> result(out_result, out_result + out_len);
	XGB_CALL(XGDMatrixFree(matrix));
	return result;
}

static void SetParam(BoosterHandle booster, const char *name, const string &value) {
	XGB_CALL(XGBoosterSetParam(booster, name, value.c_str()));
}

static void SetParam(BoosterHandle booster, const char *name, double value) {
	SetParam(booster, name, to_string(value));
}

static void SetParam(BoosterHandle booster, const char *name, idx_t value) {
	SetParam(booster, name, to_string(value));
}

} // namespace

RegressionMetricsAccumulator::RegressionMetricsAccumulator() : rng_state_(0) {
	abs_error_sample_.reserve(MEDIAN_SAMPLE_CAP);
}

void RegressionMetricsAccumulator::Update(const vector<float> &predictions, const vector<float> &labels) {
	if (predictions.size() != labels.size()) {
		throw InternalException("RegressionMetricsAccumulator received mismatched prediction and label sizes");
	}
	for (idx_t i = 0; i < predictions.size(); i++) {
		double pred = predictions[i];
		double label = labels[i];
		double err = label - pred;
		double abs_err = std::abs(err);
		double sq_err = err * err;
		double safe_label = std::max(label, 0.0);
		double safe_pred = std::max(pred, 0.0);
		double log_err = std::log1p(safe_label) - std::log1p(safe_pred);

		n_++;
		sum_abs_error_ += abs_err;
		sum_sq_error_ += sq_err;
		sum_sq_log_error_ += log_err * log_err;
		sum_y_ += label;
		sum_y2_ += label * label;
		sum_err_ += err;
		sum_err2_ += err * err;

		seen_abs_errors_++;
		if (abs_error_sample_.size() < MEDIAN_SAMPLE_CAP) {
			abs_error_sample_.push_back(abs_err);
		} else {
			rng_state_ = rng_state_ * 1664525u + 1013904223u;
			auto replace_idx = UnsafeNumericCast<idx_t>(rng_state_ % seen_abs_errors_);
			if (replace_idx < MEDIAN_SAMPLE_CAP) {
				abs_error_sample_[replace_idx] = abs_err;
			}
		}
	}
}

RegressionMetrics RegressionMetricsAccumulator::Finalize() const {
	RegressionMetrics metrics;
	if (n_ == 0) {
		return metrics;
	}

	auto n = static_cast<double>(n_);
	metrics.mean_absolute_error = sum_abs_error_ / n;
	metrics.mean_squared_error = sum_sq_error_ / n;
	metrics.mean_squared_log_error = sum_sq_log_error_ / n;

	vector<double> sample = abs_error_sample_;
	if (!sample.empty()) {
		auto mid = sample.begin() + sample.size() / 2;
		std::nth_element(sample.begin(), mid, sample.end());
		metrics.median_absolute_error = *mid;
	}

	double y_mean = sum_y_ / n;
	double ss_tot = std::max(0.0, sum_y2_ - n * y_mean * y_mean);
	if (ss_tot > 0.0) {
		metrics.r2_score = 1.0 - (sum_sq_error_ / ss_tot);
	} else {
		metrics.r2_score = 0.0;
	}

	double y_var = std::max(0.0, (sum_y2_ / n) - (y_mean * y_mean));
	double err_mean = sum_err_ / n;
	double err_var = std::max(0.0, (sum_err2_ / n) - (err_mean * err_mean));
	if (y_var > 0.0) {
		metrics.explained_variance = 1.0 - (err_var / y_var);
	} else {
		metrics.explained_variance = 0.0;
	}
	return metrics;
}

string ParseModelType(const Value &options_value) {
	if (options_value.IsNull()) {
		throw InvalidInputException("ml_fit options cannot be NULL");
	}
	if (options_value.type().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("ml_fit options must be a STRUCT");
	}
	auto options_type = options_value.type();
	auto &types = StructType::GetChildTypes(options_type);
	auto &values = StructValue::GetChildren(options_value);
	for (idx_t i = 0; i < types.size(); i++) {
		if (StringUtil::Lower(types[i].first) == "model_type") {
			if (values[i].IsNull()) {
				throw InvalidInputException("options.model_type is required");
			}
			return StringUtil::Lower(StringValue::Get(values[i]));
		}
	}
	throw InvalidInputException("options.model_type is required");
}

BoostedTreeFitOptions ParseBoostedTreeFitOptions(const Value &options_value) {
	BoostedTreeFitOptions options;
	auto model_type = ParseModelType(options_value);
	if (model_type != "boosted_tree_regressor") {
		throw InvalidInputException("ParseBoostedTreeFitOptions requires model_type='boosted_tree_regressor'");
	}

	auto options_type = options_value.type();
	auto &types = StructType::GetChildTypes(options_type);
	auto &values = StructValue::GetChildren(options_value);
	for (idx_t i = 0; i < types.size(); i++) {
		auto key = StringUtil::Lower(types[i].first);
		auto &value = values[i];
		if (value.IsNull() || key == "model_type") {
			continue;
		}
		if (key == "label") {
			options.label = StringValue::Get(value.DefaultCastAs(LogicalType::VARCHAR));
		} else if (key == "booster_type") {
			options.booster_type = StringUtil::Lower(StringValue::Get(value.DefaultCastAs(LogicalType::VARCHAR)));
		} else if (key == "dart_normalize_type") {
			options.dart_normalize_type = StringUtil::Lower(StringValue::Get(value.DefaultCastAs(LogicalType::VARCHAR)));
		} else if (key == "num_parallel_tree") {
			options.num_parallel_tree = UnsafeNumericCast<idx_t>(value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>());
		} else if (key == "max_tree_depth") {
			options.max_tree_depth = UnsafeNumericCast<idx_t>(value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>());
		} else if (key == "dropout") {
			options.dropout = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "l1_reg") {
			options.l1_reg = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "l2_reg") {
			options.l2_reg = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "learning_rate") {
			options.learning_rate = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "tree_method") {
			options.tree_method = StringUtil::Lower(StringValue::Get(value.DefaultCastAs(LogicalType::VARCHAR)));
		} else if (key == "min_child_weight") {
			options.min_child_weight = UnsafeNumericCast<idx_t>(value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>());
		} else if (key == "colsample_bytree") {
			options.colsample_bytree = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "colsample_bylevel") {
			options.colsample_bylevel = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "colsample_bynode") {
			options.colsample_bynode = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "min_split_loss") {
			options.min_split_loss = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "subsample") {
			options.subsample = DoubleValue::Get(value.DefaultCastAs(LogicalType::DOUBLE));
		} else if (key == "regression_objective") {
			options.regression_objective =
			    StringUtil::Lower(StringValue::Get(value.DefaultCastAs(LogicalType::VARCHAR)));
		} else if (key == "max_iterations") {
			options.max_iterations = UnsafeNumericCast<idx_t>(value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>());
		}
	}

	if (!StringUtil::CIEquals(options.booster_type, "gbtree") && !StringUtil::CIEquals(options.booster_type, "dart")) {
		throw InvalidInputException("options.booster_type must be one of: gbtree, dart");
	}
	if (!StringUtil::CIEquals(options.dart_normalize_type, "tree") &&
	    !StringUtil::CIEquals(options.dart_normalize_type, "forest")) {
		throw InvalidInputException("options.dart_normalize_type must be one of: tree, forest");
	}
	if (options.booster_type != "dart" && HasOptionKey(options_value, "dropout") && options.dropout > 0.0) {
		throw InvalidInputException("options.dropout requires options.booster_type='dart'");
	}
	if (options.tree_method != "hist" && options.tree_method != "approx") {
		throw InvalidInputException("options.tree_method must be one of: hist, approx");
	}
	ObjectiveForRegressionObjective(options.regression_objective);

	ValidatePositiveInteger("num_parallel_tree", options.num_parallel_tree);
	ValidatePositiveInteger("max_tree_depth", options.max_tree_depth);
	ValidatePositiveInteger("max_iterations", options.max_iterations);
	ValidateProbability("dropout", options.dropout, true);
	ValidateNonNegative("l1_reg", options.l1_reg);
	ValidateNonNegative("l2_reg", options.l2_reg);
	ValidateProbability("learning_rate", options.learning_rate, true);
	ValidateNonNegative("min_split_loss", options.min_split_loss);
	ValidateProbability("colsample_bytree", options.colsample_bytree, false);
	ValidateProbability("colsample_bylevel", options.colsample_bylevel, false);
	ValidateProbability("colsample_bynode", options.colsample_bynode, false);
	ValidateProbability("subsample", options.subsample, false);
	if (options.min_child_weight < 0) {
		throw InvalidInputException("options.min_child_weight must be >= 0");
	}

	return options;
}

bool IsBoostedTreeRegressorModelBlob(const string &blob) {
	if (blob.size() < sizeof(uint32_t) * 2) {
		return false;
	}
	uint32_t magic = 0;
	memcpy(&magic, blob.data(), sizeof(magic));
	return magic == BTR_MAGIC;
}

string SerializeBoostedTreeRegressorModel(const BoostedTreeRegressorModel &model) {
	BlobWriter writer;
	writer.WritePod(BTR_MAGIC);
	writer.WritePod(BTR_VERSION);
	writer.WriteString(model.model_type);
	writer.WriteString(model.label);
	writer.WriteString(model.booster_type);
	writer.WritePod(UnsafeNumericCast<uint64_t>(model.max_iterations));
	writer.WritePod(UnsafeNumericCast<uint64_t>(model.feature_names.size()));
	for (const auto &name : model.feature_names) {
		writer.WriteString(name);
	}
	writer.WriteString(model.xgboost_model_blob);
	return writer.blob;
}

BoostedTreeRegressorModel DeserializeBoostedTreeRegressorModel(const string &blob) {
	BlobReader reader(blob);
	if (reader.ReadPod<uint32_t>() != BTR_MAGIC) {
		throw InvalidInputException("Invalid boosted_tree_regressor model blob: magic mismatch");
	}
	auto version = reader.ReadPod<uint32_t>();
	if (version != 1 && version != BTR_VERSION) {
		throw InvalidInputException("Invalid boosted_tree_regressor model blob: unsupported version");
	}

	BoostedTreeRegressorModel model;
	model.model_type = reader.ReadString();
	model.label = reader.ReadString();
	model.booster_type = reader.ReadString();
	model.max_iterations = UnsafeNumericCast<idx_t>(reader.ReadPod<uint64_t>());
	auto n_features = reader.ReadPod<uint64_t>();
	model.feature_names.reserve(n_features);
	for (idx_t i = 0; i < n_features; i++) {
		model.feature_names.push_back(reader.ReadString());
	}
	if (version == 1) {
		reader.ReadPod<double>();
		reader.ReadPod<double>();
		reader.ReadPod<double>();
		reader.ReadPod<double>();
		reader.ReadPod<double>();
		reader.ReadPod<double>();
	}
	model.xgboost_model_blob = reader.ReadString();
	if (reader.offset != blob.size()) {
		throw InvalidInputException("Invalid boosted_tree_regressor model blob: trailing bytes");
	}
	if (model.model_type != "boosted_tree_regressor") {
		throw InvalidInputException("Invalid boosted_tree_regressor model blob: model_type mismatch");
	}
	return model;
}

void WriteBoostedTreeBatchFile(const string &path, idx_t n_rows, idx_t n_features, const vector<float> &features,
                               const vector<float> &labels) {
	if (features.size() != n_rows * n_features) {
		throw InternalException("WriteBoostedTreeBatchFile received inconsistent feature shape");
	}
	if (labels.size() != n_rows) {
		throw InternalException("WriteBoostedTreeBatchFile received inconsistent labels shape");
	}

	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	if (!out.good()) {
		throw IOException("Could not write boosted tree training batch file: %s", path);
	}
	uint64_t rows = n_rows;
	uint64_t cols = n_features;
	out.write(reinterpret_cast<const char *>(&BATCH_MAGIC), sizeof(BATCH_MAGIC));
	out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
	out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
	out.write(reinterpret_cast<const char *>(features.data()),
	          UnsafeNumericCast<std::streamsize>(features.size() * sizeof(float)));
	out.write(reinterpret_cast<const char *>(labels.data()),
	          UnsafeNumericCast<std::streamsize>(labels.size() * sizeof(float)));
	if (!out.good()) {
		throw IOException("Failed writing boosted tree training batch file: %s", path);
	}
}

BoostedTreeRegressorModel TrainBoostedTreeRegressorFromBatchFiles(const BoostedTreeFitOptions &options,
                                                                   const vector<string> &feature_names,
                                                                   const vector<string> &batch_files,
                                                                   const string &cache_prefix) {
	if (feature_names.empty()) {
		throw InvalidInputException("boosted_tree_regressor requires at least one feature column");
	}
	if (batch_files.empty()) {
		throw InvalidInputException("boosted_tree_regressor requires at least one training row");
	}

	ExternalMemoryIterator iter;
	iter.batch_files = batch_files;
	iter.n_features = feature_names.size();
	XGB_CALL(XGProxyDMatrixCreate(&iter.proxy));

	DMatrixHandle dtrain = nullptr;
	std::ostringstream matrix_config;
	matrix_config << "{\"missing\": NaN, \"cache_prefix\":\"" << cache_prefix << "\"}";
	if (options.tree_method == "hist") {
		XGB_CALL(XGExtMemQuantileDMatrixCreateFromCallback(&iter, iter.proxy, nullptr, ExternalMemoryReset, ExternalMemoryNext,
		                                                   matrix_config.str().c_str(), &dtrain));
	} else {
		XGB_CALL(XGDMatrixCreateFromCallback(&iter, iter.proxy, ExternalMemoryReset, ExternalMemoryNext,
		                                     matrix_config.str().c_str(), &dtrain));
	}

	BoosterHandle booster = nullptr;
	DMatrixHandle cache_mats[1] = {dtrain};
	XGB_CALL(XGBoosterCreate(cache_mats, 1, &booster));

	SetParam(booster, "seed", string("0"));
	SetParam(booster, "verbosity", string("0"));
	SetParam(booster, "booster", options.booster_type);
	if (options.booster_type == "dart") {
		SetParam(booster, "normalize_type", options.dart_normalize_type);
		SetParam(booster, "rate_drop", options.dropout);
	}
	SetParam(booster, "num_parallel_tree", options.num_parallel_tree);
	SetParam(booster, "max_depth", options.max_tree_depth);
	SetParam(booster, "alpha", options.l1_reg);
	SetParam(booster, "lambda", options.l2_reg);
	SetParam(booster, "eta", options.learning_rate);
	SetParam(booster, "tree_method", options.tree_method);
	SetParam(booster, "min_child_weight", options.min_child_weight);
	SetParam(booster, "colsample_bytree", options.colsample_bytree);
	SetParam(booster, "colsample_bylevel", options.colsample_bylevel);
	SetParam(booster, "colsample_bynode", options.colsample_bynode);
	SetParam(booster, "gamma", options.min_split_loss);
	SetParam(booster, "subsample", options.subsample);
	SetParam(booster, "objective", ObjectiveForRegressionObjective(options.regression_objective));

	vector<const char *> c_feature_names;
	c_feature_names.reserve(feature_names.size());
	for (const auto &feature_name : feature_names) {
		c_feature_names.push_back(feature_name.c_str());
	}
	XGB_CALL(XGBoosterSetStrFeatureInfo(booster, "feature_name", c_feature_names.data(), c_feature_names.size()));

	for (idx_t i = 0; i < options.max_iterations; i++) {
		XGB_CALL(XGBoosterUpdateOneIter(booster, i, dtrain));
	}

	bst_ulong model_len = 0;
	const char *model_bytes = nullptr;
	XGB_CALL(XGBoosterSaveModelToBuffer(booster, "{\"format\":\"ubj\"}", &model_len, &model_bytes));

	BoostedTreeRegressorModel model;
	model.model_type = "boosted_tree_regressor";
	model.label = options.label;
	model.booster_type = options.booster_type;
	model.max_iterations = options.max_iterations;
	model.feature_names = feature_names;
	model.xgboost_model_blob.assign(model_bytes, model_len);

	XGB_CALL(XGBoosterFree(booster));
	XGB_CALL(XGDMatrixFree(dtrain));
	XGB_CALL(XGDMatrixFree(iter.proxy));

	return model;
}

struct BoostedTreePredictor::Impl {
	explicit Impl(const BoostedTreeRegressorModel &model)
	    : n_features(model.feature_names.size()), max_iterations(model.max_iterations), booster_type(model.booster_type) {
		XGB_CALL(XGBoosterCreate(nullptr, 0, &booster));
		XGB_CALL(XGBoosterLoadModelFromBuffer(booster, model.xgboost_model_blob.data(), model.xgboost_model_blob.size()));
	}

	~Impl() {
		if (booster) {
			XGBoosterFree(booster);
			booster = nullptr;
		}
	}

	vector<float> Predict(const vector<float> &features, idx_t rows, idx_t cols, int predict_type) const {
		if (cols != n_features) {
			throw InvalidInputException("Feature count mismatch: model expects %llu but got %llu", n_features, cols);
		}
		return PredictDenseInternal(booster, features, rows, cols, max_iterations, predict_type);
	}

	BoosterHandle booster = nullptr;
	idx_t n_features;
	idx_t max_iterations;
	string booster_type;
};

BoostedTreePredictor::BoostedTreePredictor(const BoostedTreeRegressorModel &model) : impl_(make_uniq<Impl>(model)) {
}

BoostedTreePredictor::~BoostedTreePredictor() = default;

vector<float> BoostedTreePredictor::Predict(const vector<float> &features, idx_t rows, idx_t cols) const {
	auto predictions = impl_->Predict(features, rows, cols, 0);
	if (predictions.size() != rows) {
		throw InvalidInputException("Unexpected XGBoost prediction shape for boosted_tree_regressor");
	}
	return predictions;
}

vector<float> BoostedTreePredictor::PredictContributions(const vector<float> &features, idx_t rows, idx_t cols) const {
	auto contributions = impl_->Predict(features, rows, cols, 2);
	auto expected = rows * (cols + 1);
	if (contributions.size() != expected) {
		throw InvalidInputException("Unexpected XGBoost SHAP contribution shape for boosted_tree_regressor");
	}
	return contributions;
}

} // namespace ml
} // namespace duckdb
