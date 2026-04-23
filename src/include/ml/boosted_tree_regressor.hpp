// Copyright 2026
//
// Boosted tree regressor model and XGBoost wrappers.
//
// Algorithm: XGBoost gradient-boosted trees with external-memory training.
// Library: XGBoost C API.
// Constraints: chunked ingestion, model blob serialization, streaming metrics.

#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace ml {

struct BoostedTreeFitOptions {
	string model_type = "boosted_tree_regressor";
	string label = "label";
	string booster_type = "gbtree";
	string dart_normalize_type = "tree";
	idx_t num_parallel_tree = 1;
	idx_t max_tree_depth = 6;
	double dropout = 0.0;
	double l1_reg = 0.0;
	double l2_reg = 1.0;
	double learning_rate = 0.3;
	string tree_method = "hist";
	idx_t min_child_weight = 1;
	double colsample_bytree = 1.0;
	double colsample_bylevel = 1.0;
	double colsample_bynode = 1.0;
	double min_split_loss = 0.0;
	double subsample = 1.0;
	string regression_objective = "squarederror";
	idx_t max_iterations = 10;
};

struct RegressionMetrics {
	double mean_absolute_error = 0.0;
	double mean_squared_error = 0.0;
	double mean_squared_log_error = 0.0;
	double median_absolute_error = 0.0;
	double r2_score = 0.0;
	double explained_variance = 0.0;
};

class RegressionMetricsAccumulator {
public:
	RegressionMetricsAccumulator();

	void Update(const vector<float> &predictions, const vector<float> &labels);
	RegressionMetrics Finalize() const;

private:
	idx_t n_ = 0;
	double sum_abs_error_ = 0.0;
	double sum_sq_error_ = 0.0;
	double sum_sq_log_error_ = 0.0;
	double sum_y_ = 0.0;
	double sum_y2_ = 0.0;
	double sum_err_ = 0.0;
	double sum_err2_ = 0.0;
	idx_t seen_abs_errors_ = 0;
	vector<double> abs_error_sample_;
	uint32_t rng_state_ = 0;
};

struct BoostedTreeRegressorModel {
	string model_type = "boosted_tree_regressor";
	string label;
	string booster_type;
	idx_t max_iterations = 0;
	vector<string> feature_names;
	string xgboost_model_blob;
};

string ParseModelType(const Value &options_value);
BoostedTreeFitOptions ParseBoostedTreeFitOptions(const Value &options_value);

bool IsBoostedTreeRegressorModelBlob(const string &blob);
string SerializeBoostedTreeRegressorModel(const BoostedTreeRegressorModel &model);
BoostedTreeRegressorModel DeserializeBoostedTreeRegressorModel(const string &blob);

void WriteBoostedTreeBatchFile(const string &path, idx_t n_rows, idx_t n_features, const vector<float> &features,
                               const vector<float> &labels);

BoostedTreeRegressorModel TrainBoostedTreeRegressorFromBatchFiles(const BoostedTreeFitOptions &options,
                                                                   const vector<string> &feature_names,
                                                                   const vector<string> &batch_files,
                                                                   const string &cache_prefix);

class BoostedTreePredictor {
public:
	explicit BoostedTreePredictor(const BoostedTreeRegressorModel &model);
	~BoostedTreePredictor();

	vector<float> Predict(const vector<float> &features, idx_t rows, idx_t cols) const;
	vector<float> PredictContributions(const vector<float> &features, idx_t rows, idx_t cols) const;

private:
	struct Impl;
	unique_ptr<Impl> impl_;
};

} // namespace ml
} // namespace duckdb
