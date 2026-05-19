// Copyright 2026
//
// ml_fit table function for Incremental PCA training.
//
// Algorithm: sklearn-style IncrementalPCA partial_fit over DataChunk batches.
// Library: Eigen-backed implementation in src/algorithms/pca.cpp.
// Constraints: no full-table materialization.

#include "ml/register.hpp"

#include "ml/boosted_tree_regressor.hpp"
#include "ml/model_registry.hpp"
#include "ml/pca.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/binder.hpp"

#include <mutex>
#include <sstream>
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

static string EscapeJsonString(const string &value) {
	string escaped;
	escaped.reserve(value.size() + 8);
	for (auto ch : value) {
		switch (ch) {
		case '\\':
			escaped += "\\\\";
			break;
		case '"':
			escaped += "\\\"";
			break;
		case '\n':
			escaped += "\\n";
			break;
		case '\r':
			escaped += "\\r";
			break;
		case '\t':
			escaped += "\\t";
			break;
		default:
			escaped.push_back(ch);
			break;
		}
	}
	return escaped;
}

static string BuildModelTableFallback(TableFunctionBindInput &input) {
	vector<string> source_names = input.input_table_names;
	if (source_names.empty()) {
		source_names.reserve(input.input_table_types.size());
		for (idx_t i = 0; i < input.input_table_types.size(); i++) {
			source_names.push_back("feature_" + to_string(i + 1));
		}
	}

	std::ostringstream buffer;
	buffer << "{\"source\":null,\"columns\":[";
	for (idx_t i = 0; i < source_names.size(); i++) {
		if (i > 0) {
			buffer << ",";
		}
		buffer << "{\"name\":\"" << EscapeJsonString(source_names[i]) << "\",\"type\":\""
		       << EscapeJsonString(input.input_table_types[i].ToString()) << "\"}";
	}
	buffer << "],\"note\":\"DuckDB TABLE argument did not expose original SQL text\"}";
	return buffer.str();
}

static string TryBuildSimpleSubqueryTableArg(const ParsedExpression &table_arg_expr) {
	if (table_arg_expr.GetExpressionClass() != ExpressionClass::SUBQUERY) {
		return string();
	}
	auto &subquery_expr = table_arg_expr.Cast<SubqueryExpression>();
	if (!subquery_expr.subquery || !subquery_expr.subquery->node ||
	    subquery_expr.subquery->node->type != QueryNodeType::SELECT_NODE) {
		return string();
	}
	auto &select_node = subquery_expr.subquery->node->Cast<SelectNode>();
	if (!select_node.from_table) {
		return string();
	}
	string query = "(SELECT * FROM " + select_node.from_table->ToString();
	if (select_node.where_clause) {
		try {
			query += " WHERE " + select_node.where_clause->ToString();
		} catch (std::exception &) {
			// Keep the safe FROM-only fallback if WHERE cannot be stringified.
		}
	}
	query += ")";
	return query;
}

static string TryExtractTableArgFromCurrentQuery(TableFunctionBindInput &input, const ParsedExpression &table_arg_expr) {
	auto trim_copy = [](const string &value) {
		auto result = value;
		StringUtil::Trim(result);
		return result;
	};
	if (!input.binder) {
		return string();
	}
	auto query_location = table_arg_expr.GetQueryLocation();
	if (!query_location.IsValid()) {
		return string();
	}
	const auto &query = input.binder->context.GetCurrentQuery();
	auto start = query_location.GetIndex();
	if (start >= query.size()) {
		return string();
	}
	idx_t depth = 0;
	bool in_single_quote = false;
	bool in_double_quote = false;
	bool in_line_comment = false;
	bool in_block_comment = false;
	auto begins_with_paren = query[start] == '(';
	for (idx_t i = start; i < query.size(); i++) {
		auto c = query[i];
		auto next = i + 1 < query.size() ? query[i + 1] : '\0';
		if (in_line_comment) {
			if (c == '\n') {
				in_line_comment = false;
			}
			continue;
		}
		if (in_block_comment) {
			if (c == '*' && next == '/') {
				in_block_comment = false;
				i++;
			}
			continue;
		}
		if (in_single_quote) {
			if (c == '\\' && next != '\0') {
				i++;
				continue;
			}
			if (c == '\'') {
				in_single_quote = false;
			}
			continue;
		}
		if (in_double_quote) {
			if (c == '\\' && next != '\0') {
				i++;
				continue;
			}
			if (c == '"') {
				in_double_quote = false;
			}
			continue;
		}
		if (c == '-' && next == '-') {
			in_line_comment = true;
			i++;
			continue;
		}
		if (c == '/' && next == '*') {
			in_block_comment = true;
			i++;
			continue;
		}
		if (c == '\'') {
			in_single_quote = true;
			continue;
		}
		if (c == '"') {
			in_double_quote = true;
			continue;
		}
		if (c == '(') {
			depth++;
			continue;
		}
		if (c == ')') {
			if (depth == 0) {
				return trim_copy(query.substr(start, i - start));
			}
			depth--;
			if (depth == 0 && begins_with_paren) {
				return trim_copy(query.substr(start, i - start + 1));
			}
			continue;
		}
		if (c == ',' && depth == 0) {
			return trim_copy(query.substr(start, i - start));
		}
	}
	return trim_copy(query.substr(start));
}

static string BuildModelTable(TableFunctionBindInput &input) {
	if (input.ref.function && input.ref.function->GetExpressionClass() == ExpressionClass::FUNCTION) {
		auto &function_expr = input.ref.function->Cast<FunctionExpression>();
		idx_t table_arg_index = input.table_function.arguments.size();
		for (idx_t i = 0; i < input.table_function.arguments.size(); i++) {
			if (input.table_function.arguments[i].id() == LogicalTypeId::TABLE) {
				table_arg_index = i;
				break;
			}
		}
		if (table_arg_index < function_expr.children.size() && function_expr.children[table_arg_index]) {
			auto &table_arg_expr = *function_expr.children[table_arg_index];
			string table_arg_sql;
			try {
				table_arg_sql = table_arg_expr.ToString();
			} catch (std::exception &) {
				table_arg_sql = TryExtractTableArgFromCurrentQuery(input, table_arg_expr);
				if (table_arg_sql.empty()) {
					table_arg_sql = TryBuildSimpleSubqueryTableArg(table_arg_expr);
				}
			}
			if (!table_arg_sql.empty()) {
				return table_arg_sql;
			}
		}
	}
	return BuildModelTableFallback(input);
}

static string ReadModelName(const Value &value) {
	if (value.IsNull()) {
		throw InvalidInputException("ml_fit requires a non-NULL model name");
	}
	auto model_name = StringValue::Get(value);
	StringUtil::Trim(model_name);
	if (model_name.empty()) {
		throw InvalidInputException("ml_fit requires a non-empty model name");
	}
	return model_name;
}

struct MlFitBindData : public FunctionData {
	MlFitBindData(string model_name_p, string model_type_p, string options_text_p, bool has_transforms_p,
	              string transforms_text_p, string model_table_p, FitOptions pca_options_p,
	              BoostedTreeFitOptions boosted_options_p, vector<idx_t> feature_indices_p,
	              idx_t label_index_p, vector<string> model_feature_names_p)
	    : model_name(std::move(model_name_p)), model_type(std::move(model_type_p)),
	      options_text(std::move(options_text_p)), has_transforms(has_transforms_p),
	      transforms_text(std::move(transforms_text_p)), model_table(std::move(model_table_p)),
	      pca_options(std::move(pca_options_p)), boosted_options(std::move(boosted_options_p)),
	      feature_indices(std::move(feature_indices_p)), label_index(label_index_p),
	      model_feature_names(std::move(model_feature_names_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MlFitBindData>(model_name, model_type, options_text, has_transforms, transforms_text,
		                               model_table, pca_options, boosted_options, feature_indices,
		                               label_index, model_feature_names);
	}

	bool Equals(const FunctionData &other) const override {
		auto &rhs = other.Cast<MlFitBindData>();
		return model_name == rhs.model_name && model_type == rhs.model_type &&
		       feature_indices == rhs.feature_indices && label_index == rhs.label_index &&
		       model_feature_names == rhs.model_feature_names;
	}

	string model_name;
	string model_type;
	string options_text;
	bool has_transforms;
	string transforms_text;
	string model_table;
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
	auto model_name = ReadModelName(input.inputs[0]);
	auto model_type = ParseModelType(input.inputs[1]);

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
	auto options_text = input.inputs[1].ToString();
	auto has_transforms = input.inputs.size() >= 3 && !input.inputs[2].IsNull();
	auto transforms_text = has_transforms ? input.inputs[2].ToString() : string();
	auto model_table = BuildModelTable(input);

	if (model_type == "pca") {
		pca_options = ParseFitOptions(input.inputs[1]);
		model_feature_names = feature_names;
		for (idx_t i = 0; i < feature_names.size(); i++) {
			feature_indices.push_back(i);
		}
	} else if (model_type == "boosted_tree_regressor") {
		boosted_options = ParseBoostedTreeFitOptions(input.inputs[1]);
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

	names.emplace_back("model_name");
	names.emplace_back("model_version");
	names.emplace_back("model_blob");
	names.emplace_back("model_timestamp");
	names.emplace_back("model_options");
	names.emplace_back("model_transforms");
	names.emplace_back("model_table");
	return_types.emplace_back(LogicalType::VARCHAR);
	return_types.emplace_back(LogicalType::BIGINT);
	return_types.emplace_back(LogicalType::BLOB);
	return_types.emplace_back(LogicalType::VARCHAR);
	return_types.emplace_back(LogicalType::VARCHAR);
	return_types.emplace_back(LogicalType::VARCHAR);
	return_types.emplace_back(LogicalType::VARCHAR);
	return make_uniq<MlFitBindData>(
	    std::move(model_name), std::move(model_type), std::move(options_text), has_transforms,
	    std::move(transforms_text), std::move(model_table), std::move(pca_options),
	    std::move(boosted_options), std::move(feature_indices), label_index, std::move(model_feature_names));
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

	string trained_model_blob;
	if (global_state.model_type == "pca") {
		FlushPendingRows(global_state);
		if (!global_state.trainer->IsInitialized()) {
			throw InvalidInputException("ml_fit could not train model '%s': no valid training rows were provided",
			                            bind_data.model_name);
		} else {
			auto model = global_state.trainer->ToModel(bind_data.model_feature_names);
			auto blob = SerializePcaModel(model);
			trained_model_blob.assign(blob.data(), blob.size());
		}
	} else {
		FlushPendingBoostedTreeRows(global_state);
		if (global_state.batch_files.empty()) {
			throw InvalidInputException("ml_fit could not train model '%s': no valid training rows were provided",
			                            bind_data.model_name);
		} else {
			auto model = TrainBoostedTreeRegressorFromBatchFiles(bind_data.boosted_options,
			                                                    bind_data.model_feature_names,
			                                                    global_state.batch_files,
			                                                    global_state.cache_dir + "/xgb_cache/cache");
			auto blob = SerializeBoostedTreeRegressorModel(model);
			trained_model_blob.assign(blob.data(), blob.size());
		}
	}

	auto persisted = InsertModelRegistryEntry(bind_data.model_name, trained_model_blob, bind_data.options_text,
	                                         bind_data.has_transforms, bind_data.transforms_text,
	                                         bind_data.model_table);
	output.SetCardinality(1);
	FlatVector::GetData<string_t>(output.data[0])[0] = StringVector::AddString(output.data[0], bind_data.model_name);
	FlatVector::GetData<int64_t>(output.data[1])[0] = UnsafeNumericCast<int64_t>(persisted.version);
	FlatVector::GetData<string_t>(output.data[2])[0] =
	    StringVector::AddStringOrBlob(output.data[2], trained_model_blob.data(), trained_model_blob.size());
	FlatVector::GetData<string_t>(output.data[3])[0] = StringVector::AddString(output.data[3], persisted.timestamp);
	FlatVector::GetData<string_t>(output.data[4])[0] = StringVector::AddString(output.data[4], bind_data.options_text);
	if (bind_data.has_transforms) {
		FlatVector::GetData<string_t>(output.data[5])[0] =
		    StringVector::AddString(output.data[5], bind_data.transforms_text);
	} else {
		auto &validity = FlatVector::Validity(output.data[5]);
		validity.SetInvalid(0);
	}
	FlatVector::GetData<string_t>(output.data[6])[0] = StringVector::AddString(output.data[6], bind_data.model_table);

	global_state.emitted = true;
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

} // namespace

void RegisterMlFit(ExtensionLoader &loader) {
	vector<LogicalType> fit_args_with_transforms = {LogicalType(LogicalTypeId::VARCHAR),
	                                                LogicalType(LogicalTypeId::ANY),
	                                                LogicalType(LogicalTypeId::ANY),
	                                                LogicalType(LogicalTypeId::TABLE)};
	TableFunction fit_with_transforms("ml_fit", fit_args_with_transforms, nullptr, MlFitBind, MlFitInitGlobal,
	                                 nullptr);
	fit_with_transforms.in_out_function = MlFitFunction;
	fit_with_transforms.in_out_function_final = MlFitFinalize;
	loader.RegisterFunction(std::move(fit_with_transforms));
}

} // namespace ml
} // namespace duckdb
