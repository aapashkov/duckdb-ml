// Copyright 2026
//
// Incremental PCA model and training/evaluation helpers.
//
// Algorithm: sklearn.decomposition.IncrementalPCA-style merge updates.
// Library: Eigen (matrix math + SVD).
// Constraints: chunked updates, no full-table materialization.

#pragma once

#include "duckdb.hpp"
#include <Eigen/Dense>

namespace duckdb {
namespace ml {

struct FitOptions {
	string model_type;
	idx_t num_components = 2;
	bool whiten = false;
};

struct PcaModel {
	string model_type;
	bool whiten = false;
	idx_t n_features = 0;
	idx_t n_components = 0;
	idx_t n_samples_seen = 0;
	double training_total_explained_variance_ratio = 0.0;
	vector<string> feature_names;
	Eigen::VectorXd mean;
	Eigen::VectorXd var;
	Eigen::VectorXd singular_values;
	Eigen::VectorXd explained_variance;
	Eigen::VectorXd explained_variance_ratio;
	Eigen::MatrixXd components;
};

class IncrementalPca {
public:
	IncrementalPca(idx_t requested_components, bool whiten, idx_t n_features);

	void PartialFit(const Eigen::Ref<const Eigen::MatrixXd> &batch);
	void LoadFromModel(const PcaModel &model);
	PcaModel ToModel(const vector<string> &feature_names) const;

	bool IsInitialized() const;
	idx_t NFeatures() const;
	idx_t NComponents() const;
	idx_t NSamplesSeen() const;
	bool Whiten() const;

private:
	static void IncrementalMeanAndVar(const Eigen::Ref<const Eigen::MatrixXd> &batch, const Eigen::VectorXd &last_mean,
	                                  const Eigen::VectorXd &last_var, idx_t last_sample_count,
	                                  Eigen::VectorXd &new_mean, Eigen::VectorXd &new_var,
	                                  idx_t &new_sample_count);
	static void FlipSigns(Eigen::MatrixXd &v_t);

private:
	idx_t requested_components_;
	bool whiten_;
	idx_t n_features_;
	bool initialized_ = false;
	idx_t n_samples_seen_ = 0;
	Eigen::VectorXd mean_;
	Eigen::VectorXd var_;
	Eigen::MatrixXd components_;
	Eigen::VectorXd singular_values_;
	Eigen::VectorXd explained_variance_;
	Eigen::VectorXd explained_variance_ratio_;
};

class RunningMoments {
public:
	explicit RunningMoments(idx_t dimensions);

	void Update(const Eigen::Ref<const Eigen::MatrixXd> &batch);
	double TotalSampleVariance() const;
	idx_t SampleCount() const;

private:
	idx_t n_;
	Eigen::VectorXd mean_;
	Eigen::VectorXd m2_;
};

FitOptions ParseFitOptions(const Value &options_value);
string SerializePcaModel(const PcaModel &model);
PcaModel DeserializePcaModel(const string &blob);

Eigen::MatrixXd Transform(const PcaModel &model, const Eigen::Ref<const Eigen::MatrixXd> &batch);

} // namespace ml
} // namespace duckdb
