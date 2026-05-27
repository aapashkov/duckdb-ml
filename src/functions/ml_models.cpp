// Copyright 2026
//
// ml_models table function for listing persisted sqlite registry rows.

#include "ml/register.hpp"

#include "ml/model_registry.hpp"

#include "duckdb.hpp"

namespace duckdb {
namespace ml {
namespace {

struct MlModelsGlobalState : public GlobalTableFunctionState {
	MlModelsGlobalState() : rows(LoadAllModelRegistryEntries()) {
	}

	vector<RegistryModelEntry> rows;
	idx_t offset = 0;
};

static unique_ptr<FunctionData> MlModelsBind(ClientContext &, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	if (!input.inputs.empty()) {
		throw InvalidInputException("ml_models takes no positional arguments");
	}
	if (!input.named_parameters.empty()) {
		throw InvalidInputException("ml_models takes no named parameters");
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
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> MlModelsInitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<MlModelsGlobalState>();
}

static void MlModelsFunction(ClientContext &, TableFunctionInput &input_data, DataChunk &output) {
	auto &state = input_data.global_state->Cast<MlModelsGlobalState>();
	if (state.offset >= state.rows.size()) {
		output.SetCardinality(0);
		return;
	}

	auto remaining = state.rows.size() - state.offset;
	auto count = remaining < STANDARD_VECTOR_SIZE ? remaining : STANDARD_VECTOR_SIZE;
	output.SetCardinality(count);
	auto &transforms_validity = FlatVector::Validity(output.data[5]);
	auto &table_validity = FlatVector::Validity(output.data[6]);

	for (idx_t row = 0; row < count; row++) {
		auto &entry = state.rows[state.offset + row];
		FlatVector::GetData<string_t>(output.data[0])[row] = StringVector::AddString(output.data[0], entry.model_name);
		FlatVector::GetData<int64_t>(output.data[1])[row] = UnsafeNumericCast<int64_t>(entry.model_version);
		FlatVector::GetData<string_t>(output.data[2])[row] =
		    StringVector::AddStringOrBlob(output.data[2], entry.model_blob.data(), entry.model_blob.size());
		FlatVector::GetData<string_t>(output.data[3])[row] =
		    StringVector::AddString(output.data[3], entry.model_timestamp);
		FlatVector::GetData<string_t>(output.data[4])[row] =
		    StringVector::AddString(output.data[4], entry.model_options);
		if (entry.has_model_transforms) {
			FlatVector::GetData<string_t>(output.data[5])[row] =
			    StringVector::AddString(output.data[5], entry.model_transforms);
		} else {
			transforms_validity.SetInvalid(row);
		}
		if (entry.has_model_table) {
			FlatVector::GetData<string_t>(output.data[6])[row] =
			    StringVector::AddString(output.data[6], entry.model_table);
		} else {
			table_validity.SetInvalid(row);
		}
	}
	state.offset += count;
}

} // namespace

void RegisterMlModels(ExtensionLoader &loader) {
	vector<LogicalType> args;
	TableFunction models("ml_models", args, MlModelsFunction, MlModelsBind, MlModelsInitGlobal, nullptr);
	loader.RegisterFunction(std::move(models));
}

} // namespace ml
} // namespace duckdb
