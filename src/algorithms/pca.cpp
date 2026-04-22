// Copyright 2026
//
// Incremental PCA implementation and model serialization.
//
// Algorithm: sklearn.decomposition.IncrementalPCA partial_fit merge SVD.
// Library: Eigen (SVD and matrix ops).
// Constraints: batched updates, no full-table materialization.

#include "ml/pca.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace duckdb {
namespace ml {
namespace {

static constexpr uint32_t PCA_MAGIC = 0x4d4c5043; // MLPC
static constexpr uint32_t PCA_VERSION = 1;

struct BlobWriter {
	template <class T>
	void WritePod(const T &value) {
		auto start = blob.size();
		blob.resize(start + sizeof(T));
		memcpy(&blob[start], &value, sizeof(T));
	}

	void WriteString(const string &value) {
		uint64_t len = value.size();
		WritePod(len);
		blob.append(value);
	}

	void WriteVector(const Eigen::VectorXd &vec) {
		uint64_t len = vec.size();
		WritePod(len);
		for (idx_t i = 0; i < vec.size(); i++) {
			WritePod(vec[i]);
		}
	}

	void WriteMatrix(const Eigen::MatrixXd &mat) {
		uint64_t rows = mat.rows();
		uint64_t cols = mat.cols();
		WritePod(rows);
		WritePod(cols);
		for (idx_t i = 0; i < mat.rows(); i++) {
			for (idx_t j = 0; j < mat.cols(); j++) {
				WritePod(mat(i, j));
			}
		}
	}

	string blob;
};

struct BlobReader {
	explicit BlobReader(const string &blob) : blob(blob), offset(0) {
	}

	template <class T>
	T ReadPod() {
		if (offset + sizeof(T) > blob.size()) {
			throw InvalidInputException("Invalid PCA model blob: truncated payload");
		}
		T value;
		memcpy(&value, blob.data() + offset, sizeof(T));
		offset += sizeof(T);
		return value;
	}

	string ReadString() {
		auto len = ReadPod<uint64_t>();
		if (offset + len > blob.size()) {
			throw InvalidInputException("Invalid PCA model blob: invalid string length");
		}
		string result(blob.data() + offset, len);
		offset += len;
		return result;
	}

	Eigen::VectorXd ReadVector() {
		auto len = ReadPod<uint64_t>();
		Eigen::VectorXd result(static_cast<int64_t>(len));
		for (uint64_t i = 0; i < len; i++) {
			result(static_cast<int64_t>(i)) = ReadPod<double>();
		}
		return result;
	}

	Eigen::MatrixXd ReadMatrix() {
		auto rows = ReadPod<uint64_t>();
		auto cols = ReadPod<uint64_t>();
		Eigen::MatrixXd result(static_cast<int64_t>(rows), static_cast<int64_t>(cols));
		for (uint64_t i = 0; i < rows; i++) {
			for (uint64_t j = 0; j < cols; j++) {
				result(static_cast<int64_t>(i), static_cast<int64_t>(j)) = ReadPod<double>();
			}
		}
		return result;
	}

	const string &blob;
	idx_t offset;
};

} // namespace

IncrementalPca::IncrementalPca(idx_t requested_components, bool whiten, idx_t n_features)
    : requested_components_(requested_components), whiten_(whiten), n_features_(n_features), mean_(n_features),
      var_(n_features) {
	if (n_features_ == 0) {
		throw InvalidInputException("PCA requires at least one feature column");
	}
	if (requested_components_ == 0) {
		throw InvalidInputException("Option 'num_components' must be >= 1");
	}
}

void IncrementalPca::IncrementalMeanAndVar(const Eigen::Ref<const Eigen::MatrixXd> &batch,
                                           const Eigen::VectorXd &last_mean, const Eigen::VectorXd &last_var,
                                           idx_t last_sample_count, Eigen::VectorXd &new_mean, Eigen::VectorXd &new_var,
                                           idx_t &new_sample_count) {
	auto batch_rows = UnsafeNumericCast<idx_t>(batch.rows());
	new_sample_count = last_sample_count + batch_rows;
	Eigen::VectorXd batch_mean = batch.colwise().mean();
	Eigen::VectorXd centered = batch.rowwise() - batch_mean.transpose();
	Eigen::VectorXd batch_var = (centered.array().square().colwise().sum() / static_cast<double>(batch_rows)).matrix();

	if (last_sample_count == 0) {
		new_mean = batch_mean;
		new_var = batch_var;
		return;
	}

	Eigen::VectorXd delta = batch_mean - last_mean;
	new_mean = last_mean + delta * (static_cast<double>(batch_rows) / static_cast<double>(new_sample_count));
	Eigen::VectorXd m_a = last_var * static_cast<double>(last_sample_count);
	Eigen::VectorXd m_b = batch_var * static_cast<double>(batch_rows);
	Eigen::VectorXd correction =
	    delta.array().square() * (static_cast<double>(last_sample_count) * static_cast<double>(batch_rows) /
	                               static_cast<double>(new_sample_count));
	new_var = (m_a + m_b + correction).array() / static_cast<double>(new_sample_count);
}

void IncrementalPca::FlipSigns(Eigen::MatrixXd &v_t) {
	for (idx_t row = 0; row < UnsafeNumericCast<idx_t>(v_t.rows()); row++) {
		idx_t max_idx = 0;
		double max_abs = std::abs(v_t(row, 0));
		for (idx_t col = 1; col < UnsafeNumericCast<idx_t>(v_t.cols()); col++) {
			auto val = std::abs(v_t(row, UnsafeNumericCast<int64_t>(col)));
			if (val > max_abs) {
				max_abs = val;
				max_idx = col;
			}
		}
		if (v_t(row, UnsafeNumericCast<int64_t>(max_idx)) < 0) {
			v_t.row(UnsafeNumericCast<int64_t>(row)) *= -1.0;
		}
	}
}

void IncrementalPca::PartialFit(const Eigen::Ref<const Eigen::MatrixXd> &batch) {
	auto batch_rows = UnsafeNumericCast<idx_t>(batch.rows());
	if (batch_rows == 0) {
		return;
	}
	if (UnsafeNumericCast<idx_t>(batch.cols()) != n_features_) {
		throw InvalidInputException("Feature count mismatch in partial_fit");
	}

	Eigen::VectorXd new_mean;
	Eigen::VectorXd new_var;
	idx_t new_sample_count = 0;
	IncrementalMeanAndVar(batch, initialized_ ? mean_ : Eigen::VectorXd(), initialized_ ? var_ : Eigen::VectorXd(),
	                      n_samples_seen_, new_mean, new_var, new_sample_count);

	Eigen::MatrixXd stacked;
	if (!initialized_) {
		Eigen::MatrixXd centered = batch.rowwise() - new_mean.transpose();
		stacked = std::move(centered);
	} else {
		Eigen::VectorXd batch_mean = batch.colwise().mean();
		Eigen::MatrixXd centered_batch = batch.rowwise() - batch_mean.transpose();
		Eigen::MatrixXd previous = singular_values_.asDiagonal() * components_;
		Eigen::RowVectorXd correction =
		    std::sqrt((static_cast<double>(n_samples_seen_) / static_cast<double>(new_sample_count)) *
		              static_cast<double>(batch_rows)) *
		    (mean_ - batch_mean).transpose();
		stacked.resize(previous.rows() + centered_batch.rows() + 1, n_features_);
		stacked.topRows(previous.rows()) = previous;
		stacked.middleRows(previous.rows(), centered_batch.rows()) = centered_batch;
		stacked.bottomRows(1) = correction;
	}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(stacked, Eigen::ComputeThinV);
	Eigen::VectorXd singular_values = svd.singularValues();
	Eigen::MatrixXd v_t = svd.matrixV().transpose();
	FlipSigns(v_t);

	auto max_components = std::min<idx_t>({requested_components_, n_features_, UnsafeNumericCast<idx_t>(singular_values.size())});
	components_ = v_t.topRows(max_components);
	singular_values_ = singular_values.head(max_components);

	Eigen::VectorXd explained_variance_all = Eigen::VectorXd::Zero(UnsafeNumericCast<int64_t>(singular_values.size()));
	if (new_sample_count > 1) {
		explained_variance_all = singular_values.array().square() / static_cast<double>(new_sample_count - 1);
	}
	Eigen::VectorXd explained_variance_ratio_all = Eigen::VectorXd::Zero(UnsafeNumericCast<int64_t>(singular_values.size()));
	double total_var = new_var.sum() * static_cast<double>(new_sample_count);
	if (total_var > 0) {
		explained_variance_ratio_all = singular_values.array().square() / total_var;
	}

	explained_variance_ = explained_variance_all.head(max_components);
	explained_variance_ratio_ = explained_variance_ratio_all.head(max_components);
	mean_ = std::move(new_mean);
	var_ = std::move(new_var);
	n_samples_seen_ = new_sample_count;
	initialized_ = true;
}

void IncrementalPca::LoadFromModel(const PcaModel &model) {
	if (model.model_type != "pca") {
		throw InvalidInputException("Cannot load non-PCA model into IncrementalPca");
	}
	if (model.n_features != n_features_) {
		throw InvalidInputException("Cannot load PCA model with mismatched feature count");
	}
	components_ = model.components;
	singular_values_ = model.singular_values;
	explained_variance_ = model.explained_variance;
	explained_variance_ratio_ = model.explained_variance_ratio;
	mean_ = model.mean;
	var_ = model.var;
	n_samples_seen_ = model.n_samples_seen;
	initialized_ = true;
}

PcaModel IncrementalPca::ToModel(const vector<string> &feature_names) const {
	if (!initialized_) {
		throw InvalidInputException("PCA model has no training data");
	}
	PcaModel model;
	model.model_type = "pca";
	model.whiten = whiten_;
	model.n_features = n_features_;
	model.n_components = UnsafeNumericCast<idx_t>(components_.rows());
	model.n_samples_seen = n_samples_seen_;
	model.training_total_explained_variance_ratio = explained_variance_ratio_.sum();
	model.feature_names = feature_names;
	model.mean = mean_;
	model.var = var_;
	model.singular_values = singular_values_;
	model.explained_variance = explained_variance_;
	model.explained_variance_ratio = explained_variance_ratio_;
	model.components = components_;
	return model;
}

bool IncrementalPca::IsInitialized() const {
	return initialized_;
}

idx_t IncrementalPca::NFeatures() const {
	return n_features_;
}

idx_t IncrementalPca::NComponents() const {
	return UnsafeNumericCast<idx_t>(components_.rows());
}

idx_t IncrementalPca::NSamplesSeen() const {
	return n_samples_seen_;
}

bool IncrementalPca::Whiten() const {
	return whiten_;
}

RunningMoments::RunningMoments(idx_t dimensions)
    : n_(0), mean_(Eigen::VectorXd::Zero(UnsafeNumericCast<int64_t>(dimensions))),
      m2_(Eigen::VectorXd::Zero(UnsafeNumericCast<int64_t>(dimensions))) {
}

void RunningMoments::Update(const Eigen::Ref<const Eigen::MatrixXd> &batch) {
	for (idx_t i = 0; i < UnsafeNumericCast<idx_t>(batch.rows()); i++) {
		n_++;
		Eigen::VectorXd x = batch.row(UnsafeNumericCast<int64_t>(i)).transpose();
		Eigen::VectorXd delta = x - mean_;
		mean_ += delta / static_cast<double>(n_);
		Eigen::VectorXd delta2 = x - mean_;
		m2_ += delta.cwiseProduct(delta2);
	}
}

double RunningMoments::TotalSampleVariance() const {
	if (n_ < 2) {
		return 0.0;
	}
	return (m2_.array() / static_cast<double>(n_ - 1)).sum();
}

idx_t RunningMoments::SampleCount() const {
	return n_;
}

FitOptions ParseFitOptions(const Value &options_value) {
	if (options_value.IsNull()) {
		throw InvalidInputException("ml_fit options cannot be NULL");
	}
	if (options_value.type().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("ml_fit options must be a STRUCT");
	}
	FitOptions result;
	result.num_components = 2;
	result.whiten = false;

	auto &types = StructType::GetChildTypes(options_value.type());
	auto &values = StructValue::GetChildren(options_value);
	bool has_model_type = false;
	for (idx_t i = 0; i < types.size(); i++) {
		auto key = StringUtil::Lower(types[i].first);
		const auto &value = values[i];
		if (key == "model_type") {
			if (value.IsNull()) {
				throw InvalidInputException("options.model_type is required");
			}
			result.model_type = StringUtil::Lower(StringValue::Get(value));
			has_model_type = true;
		} else if (key == "num_components") {
			if (!value.IsNull()) {
				auto components = value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
				if (components <= 0) {
					throw InvalidInputException("options.num_components must be >= 1");
				}
				result.num_components = UnsafeNumericCast<idx_t>(components);
			}
		} else if (key == "whiten") {
			if (!value.IsNull()) {
				result.whiten = BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
			}
		}
	}

	if (!has_model_type) {
		throw InvalidInputException("options.model_type is required");
	}
	if (result.model_type != "pca") {
		throw NotImplementedException("Unsupported model_type '%s'", result.model_type);
	}
	return result;
}

string SerializePcaModel(const PcaModel &model) {
	BlobWriter writer;
	writer.WritePod(PCA_MAGIC);
	writer.WritePod(PCA_VERSION);
	writer.WriteString(model.model_type);
	writer.WritePod(static_cast<uint8_t>(model.whiten));
	writer.WritePod(static_cast<uint64_t>(model.n_features));
	writer.WritePod(static_cast<uint64_t>(model.n_components));
	writer.WritePod(static_cast<uint64_t>(model.n_samples_seen));
	writer.WritePod(model.training_total_explained_variance_ratio);
	writer.WritePod(static_cast<uint64_t>(model.feature_names.size()));
	for (const auto &name : model.feature_names) {
		writer.WriteString(name);
	}
	writer.WriteVector(model.mean);
	writer.WriteVector(model.var);
	writer.WriteVector(model.singular_values);
	writer.WriteVector(model.explained_variance);
	writer.WriteVector(model.explained_variance_ratio);
	writer.WriteMatrix(model.components);
	return writer.blob;
}

PcaModel DeserializePcaModel(const string &blob) {
	BlobReader reader(blob);
	if (reader.ReadPod<uint32_t>() != PCA_MAGIC) {
		throw InvalidInputException("Invalid PCA model blob: magic mismatch");
	}
	if (reader.ReadPod<uint32_t>() != PCA_VERSION) {
		throw InvalidInputException("Invalid PCA model blob: unsupported version");
	}
	PcaModel model;
	model.model_type = reader.ReadString();
	model.whiten = reader.ReadPod<uint8_t>() != 0;
	model.n_features = UnsafeNumericCast<idx_t>(reader.ReadPod<uint64_t>());
	model.n_components = UnsafeNumericCast<idx_t>(reader.ReadPod<uint64_t>());
	model.n_samples_seen = UnsafeNumericCast<idx_t>(reader.ReadPod<uint64_t>());
	model.training_total_explained_variance_ratio = reader.ReadPod<double>();
	auto feature_count = reader.ReadPod<uint64_t>();
	model.feature_names.reserve(feature_count);
	for (uint64_t i = 0; i < feature_count; i++) {
		model.feature_names.push_back(reader.ReadString());
	}
	model.mean = reader.ReadVector();
	model.var = reader.ReadVector();
	model.singular_values = reader.ReadVector();
	model.explained_variance = reader.ReadVector();
	model.explained_variance_ratio = reader.ReadVector();
	model.components = reader.ReadMatrix();
	if (reader.offset != blob.size()) {
		throw InvalidInputException("Invalid PCA model blob: trailing bytes");
	}
	return model;
}

Eigen::MatrixXd Transform(const PcaModel &model, const Eigen::Ref<const Eigen::MatrixXd> &batch) {
	if (UnsafeNumericCast<idx_t>(batch.cols()) != model.n_features) {
		throw InvalidInputException("ml_predict feature count mismatch");
	}
	Eigen::MatrixXd centered = batch.rowwise() - model.mean.transpose();
	Eigen::MatrixXd transformed = centered * model.components.transpose();
	if (model.whiten) {
		for (idx_t i = 0; i < model.n_components; i++) {
			double denom = std::sqrt(model.explained_variance(static_cast<int64_t>(i)));
			if (denom > 0) {
				transformed.col(static_cast<int64_t>(i)) /= denom;
			} else {
				transformed.col(static_cast<int64_t>(i)).setZero();
			}
		}
	}
	return transformed;
}

} // namespace ml
} // namespace duckdb
