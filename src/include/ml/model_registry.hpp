// Copyright 2026
//
// SQLite-backed model registry for persisted model blobs.

#pragma once

#include "duckdb/common/types.hpp"

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

const string &GetModelRegistryTableName();
string GetModelRegistryPath();

RegistryInsertResult InsertModelRegistryEntry(const string &model_name, const string &blob,
                                              const string &table_provenance,
                                              const string &options_text, bool has_transforms,
                                              const string &transforms_text);
RegistryLookupResult LoadModelRegistryEntry(const string &model_name, idx_t version);

} // namespace ml
} // namespace duckdb
