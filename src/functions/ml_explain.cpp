// Copyright 2026
//
// ml_explain placeholder for PCA.
//
// PCA does not provide per-row feature attribution in this extension,
// so this function is registered and fails with a user-facing error.

#include "ml/register.hpp"

#include "ml/pca.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {
namespace ml {
namespace {

static unique_ptr<FunctionData> MlExplainBind(ClientContext &, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("ml_explain requires a non-NULL model blob");
	}
	auto model = DeserializePcaModel(StringValue::Get(input.inputs[0]));
	if (model.model_type != "pca") {
		throw NotImplementedException("ml_explain is not supported for model_type '%s'", model.model_type);
	}
	names.emplace_back("feature_attribution");
	return_types.emplace_back(LogicalType::DOUBLE);
	return make_uniq<TableFunctionData>();
}

static OperatorResultType MlExplainFunction(ExecutionContext &, TableFunctionInput &, DataChunk &, DataChunk &) {
	throw InvalidInputException(
	    "ml_explain is not supported for model_type 'pca': feature attribution is undefined for PCA");
}

} // namespace

void RegisterMlExplain(ExtensionLoader &loader) {
	TableFunction explain_fn("ml_explain", {LogicalType::BLOB, LogicalType::TABLE}, nullptr, MlExplainBind);
	explain_fn.in_out_function = MlExplainFunction;
	loader.RegisterFunction(std::move(explain_fn));
}

} // namespace ml
} // namespace duckdb
