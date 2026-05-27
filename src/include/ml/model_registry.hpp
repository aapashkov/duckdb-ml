// Copyright 2026
//
// SQLite-backed model registry for persisted model blobs.

#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace ml {

struct RegistryInsertResult {
	idx_t version;
	string timestamp;
};

struct RegistryLookupResult {
	string blob;
	idx_t version;
	string timestamp;
};

struct RegistryModelEntry {
	string model_name;
	idx_t model_version;
	string model_blob;
	string model_timestamp;
	string model_options;
	bool has_model_transforms;
	string model_transforms;
	bool has_model_table;
	string model_table;
};

const string &GetModelRegistryTableName();
string GetModelRegistryPath();

RegistryInsertResult InsertModelRegistryEntry(const string &model_name, const string &blob,
	                                          const string &options_text, bool has_transforms,
	                                          const string &transforms_text,
	                                          const string &model_table);
RegistryLookupResult LoadModelRegistryEntry(const string &model_name, idx_t version);
vector<RegistryModelEntry> LoadAllModelRegistryEntries();

} // namespace ml
} // namespace duckdb
