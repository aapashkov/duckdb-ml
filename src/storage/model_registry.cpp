// Copyright 2026
//
// SQLite-backed model registry implementation.

#include "ml/model_registry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"

#include <cstdlib>
#include <unordered_map>

#include <sqlite3.h>

namespace duckdb {
namespace ml {
namespace {

static constexpr const char *REGISTRY_TABLE_NAME = "duckdb_ml_models";

string JoinPath(const string &base, const string &name) {
	auto fs = FileSystem::CreateLocal();
	return fs->JoinPath(base, name);
}

string ExtractParentPath(const string &path) {
	auto pos = path.find_last_of("/\\");
	if (pos == string::npos) {
		return string();
	}
	if (pos == 0) {
		return path.substr(0, 1);
	}
	return path.substr(0, pos);
}

string ResolveDefaultRegistryPath() {
#ifdef _WIN32
	auto *appdata = std::getenv("APPDATA");
	if (appdata && *appdata) {
		return JoinPath(JoinPath(string(appdata), "duckdb-ml"), "duckdb-ml.db");
	}
	auto *userprofile = std::getenv("USERPROFILE");
	if (userprofile && *userprofile) {
		return JoinPath(string(userprofile), ".duckdb-ml.db");
	}
	return JoinPath(".", ".duckdb-ml.db");
#else
	auto *home = std::getenv("HOME");
	if (home && *home) {
		return JoinPath(string(home), ".duckdb-ml.db");
	}
	return JoinPath(".", ".duckdb-ml.db");
#endif
}

void EnsureParentDirectoryExists(const string &path) {
	auto parent = ExtractParentPath(path);
	if (parent.empty()) {
		return;
	}
	auto fs = FileSystem::CreateLocal();
	if (!fs->DirectoryExists(parent)) {
		fs->CreateDirectoriesRecursive(parent);
	}
}

string ResolveRegistryPath() {
	auto *project_db_path = std::getenv("PROJECT_DB_PATH");
	if (project_db_path && *project_db_path) {
		return string(project_db_path);
	}
	auto default_path = ResolveDefaultRegistryPath();
	EnsureParentDirectoryExists(default_path);
	return default_path;
}

class SqliteStatement {
public:
	SqliteStatement(sqlite3 *db, const string &sql) {
		if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
			throw IOException("SQLite prepare failed: %s", sqlite3_errmsg(db));
		}
	}

	~SqliteStatement() {
		if (stmt) {
			sqlite3_finalize(stmt);
		}
	}

	sqlite3_stmt *Get() {
		return stmt;
	}

private:
	sqlite3_stmt *stmt = nullptr;
};

class SqliteConnection {
public:
	explicit SqliteConnection(string path_p) : path(std::move(path_p)) {
		if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) != SQLITE_OK) {
			string error = db ? sqlite3_errmsg(db) : "unknown error";
			if (db) {
				sqlite3_close(db);
				db = nullptr;
			}
			throw IOException("Could not open model registry database '%s': %s", path, error);
		}
	}

	~SqliteConnection() {
		if (db) {
			sqlite3_close(db);
		}
	}

	sqlite3 *Get() {
		return db;
	}

	const string &Path() const {
		return path;
	}

private:
	string path;
	sqlite3 *db = nullptr;
};

void ExecSql(sqlite3 *db, const string &sql) {
	char *error = nullptr;
	if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error) != SQLITE_OK) {
		string message = error ? error : "unknown error";
		sqlite3_free(error);
		throw IOException("SQLite execution failed: %s", message);
	}
}

bool TableExists(sqlite3 *db) {
	SqliteStatement stmt(db, "SELECT name FROM sqlite_master WHERE type='table' AND name=?1");
	sqlite3_bind_text(stmt.Get(), 1, REGISTRY_TABLE_NAME, -1, SQLITE_STATIC);
	auto result = sqlite3_step(stmt.Get());
	if (result == SQLITE_ROW) {
		return true;
	}
	if (result != SQLITE_DONE) {
		throw IOException("SQLite table lookup failed: %s", sqlite3_errmsg(db));
	}
	return false;
}

void ValidateExistingTableSchema(sqlite3 *db, const string &path) {
	SqliteStatement stmt(db, "PRAGMA table_info(duckdb_ml_models)");
	struct ColumnDef {
		string type;
		bool not_null;
		idx_t pk;
	};
	std::unordered_map<string, ColumnDef> expected = {
	    {"model", {"TEXT", true, 1}},         {"version", {"INTEGER", true, 2}},
	    {"blob", {"BLOB", true, 0}},          {"timestamp", {"TEXT", true, 0}},
	    {"table", {"TEXT", false, 0}},        {"options", {"TEXT", true, 0}},
	    {"transforms", {"TEXT", false, 0}},
	};
	idx_t seen_count = 0;
	while (true) {
		auto rc = sqlite3_step(stmt.Get());
		if (rc == SQLITE_DONE) {
			break;
		}
		if (rc != SQLITE_ROW) {
			throw IOException("SQLite schema inspection failed: %s", sqlite3_errmsg(db));
		}
		auto *name_text = reinterpret_cast<const char *>(sqlite3_column_text(stmt.Get(), 1));
		auto *type_text = reinterpret_cast<const char *>(sqlite3_column_text(stmt.Get(), 2));
		if (!name_text || !type_text) {
			throw InvalidInputException("Model registry table '%s' in '%s' has an incompatible schema",
			                            REGISTRY_TABLE_NAME, path);
		}
		string name(name_text);
		auto found = expected.find(name);
		if (found == expected.end()) {
			throw InvalidInputException(
			    "Model registry table '%s' in '%s' has unexpected column '%s'; use a different PROJECT_DB_PATH",
			    REGISTRY_TABLE_NAME, path, name);
		}
		auto actual_type = StringUtil::Upper(string(type_text));
		auto actual_not_null = sqlite3_column_int(stmt.Get(), 3) != 0;
		auto actual_pk = UnsafeNumericCast<idx_t>(sqlite3_column_int(stmt.Get(), 5));
		if (actual_type != found->second.type || actual_not_null != found->second.not_null ||
		    actual_pk != found->second.pk) {
			throw InvalidInputException("Model registry table '%s' in '%s' has incompatible column '%s'",
			                            REGISTRY_TABLE_NAME, path, name);
		}
		expected.erase(found);
		seen_count++;
	}
	if (!expected.empty() || seen_count != 7) {
		throw InvalidInputException(
		    "Model registry table '%s' in '%s' has an incompatible schema; expected extension-owned registry columns",
		    REGISTRY_TABLE_NAME, path);
	}
}

void EnsureRegistrySchema(SqliteConnection &connection) {
	auto *db = connection.Get();
	if (TableExists(db)) {
		ValidateExistingTableSchema(db, connection.Path());
	} else {
		ExecSql(db,
		        "CREATE TABLE IF NOT EXISTS duckdb_ml_models ("
		        "model TEXT NOT NULL,"
		        "version INTEGER NOT NULL CHECK (version > 0),"
		        "blob BLOB NOT NULL,"
		        "timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
		        "\"table\" TEXT,"
		        "options TEXT NOT NULL,"
		        "transforms TEXT,"
		        "PRIMARY KEY (model, version)"
		        ")");
	}
	ExecSql(db, "CREATE INDEX IF NOT EXISTS idx_duckdb_ml_models_latest ON duckdb_ml_models(model, version DESC)");
}

idx_t NextModelVersion(sqlite3 *db, const string &model_name) {
	SqliteStatement stmt(db, "SELECT COALESCE(MAX(version), 0) + 1 FROM duckdb_ml_models WHERE model = ?1");
	sqlite3_bind_text(stmt.Get(), 1, model_name.c_str(), -1, SQLITE_TRANSIENT);
	auto rc = sqlite3_step(stmt.Get());
	if (rc != SQLITE_ROW) {
		throw IOException("Could not allocate model version: %s", sqlite3_errmsg(db));
	}
	return UnsafeNumericCast<idx_t>(sqlite3_column_int64(stmt.Get(), 0));
}

string LoadTimestamp(sqlite3 *db, const string &model_name, idx_t version) {
	SqliteStatement stmt(db, "SELECT timestamp FROM duckdb_ml_models WHERE model = ?1 AND version = ?2");
	sqlite3_bind_text(stmt.Get(), 1, model_name.c_str(), -1, SQLITE_TRANSIENT);
	sqlite3_bind_int64(stmt.Get(), 2, UnsafeNumericCast<sqlite3_int64>(version));
	auto rc = sqlite3_step(stmt.Get());
	if (rc != SQLITE_ROW) {
		throw IOException("Could not read model timestamp: %s", sqlite3_errmsg(db));
	}
	auto *timestamp_text = reinterpret_cast<const char *>(sqlite3_column_text(stmt.Get(), 0));
	if (!timestamp_text) {
		return string();
	}
	return string(timestamp_text);
}

} // namespace

const string &GetModelRegistryTableName() {
	static const string table_name(REGISTRY_TABLE_NAME);
	return table_name;
}

string GetModelRegistryPath() {
	return ResolveRegistryPath();
}

RegistryInsertResult InsertModelRegistryEntry(const string &model_name, const string &blob,
                                              const string &table_provenance,
                                              const string &options_text, bool has_transforms,
                                              const string &transforms_text) {
	SqliteConnection connection(GetModelRegistryPath());
	EnsureRegistrySchema(connection);
	auto *db = connection.Get();
	bool transaction_started = false;
	try {
		ExecSql(db, "BEGIN IMMEDIATE");
		transaction_started = true;
		auto version = NextModelVersion(db, model_name);

		SqliteStatement insert_stmt(
		    db,
		    "INSERT INTO duckdb_ml_models (model, version, blob, timestamp, \"table\", options, transforms) "
		    "VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?4, ?5, ?6)");
		sqlite3_bind_text(insert_stmt.Get(), 1, model_name.c_str(), -1, SQLITE_TRANSIENT);
		sqlite3_bind_int64(insert_stmt.Get(), 2, UnsafeNumericCast<sqlite3_int64>(version));
		sqlite3_bind_blob(insert_stmt.Get(), 3, blob.data(), UnsafeNumericCast<int>(blob.size()), SQLITE_TRANSIENT);
		sqlite3_bind_text(insert_stmt.Get(), 4, table_provenance.c_str(), -1, SQLITE_TRANSIENT);
		sqlite3_bind_text(insert_stmt.Get(), 5, options_text.c_str(), -1, SQLITE_TRANSIENT);
		if (has_transforms) {
			sqlite3_bind_text(insert_stmt.Get(), 6, transforms_text.c_str(), -1, SQLITE_TRANSIENT);
		} else {
			sqlite3_bind_null(insert_stmt.Get(), 6);
		}

		auto rc = sqlite3_step(insert_stmt.Get());
		if (rc != SQLITE_DONE) {
			throw IOException("Could not persist model '%s' in '%s': %s", model_name, connection.Path(),
			                 sqlite3_errmsg(db));
		}

		ExecSql(db, "COMMIT");
		transaction_started = false;
		return {version, LoadTimestamp(db, model_name, version)};
	} catch (...) {
		if (transaction_started) {
			try {
				ExecSql(db, "ROLLBACK");
			} catch (...) {
			}
		}
		throw;
	}
}

RegistryLookupResult LoadModelRegistryEntry(const string &model_name, idx_t version) {
	SqliteConnection connection(GetModelRegistryPath());
	EnsureRegistrySchema(connection);
	auto *db = connection.Get();

	string sql;
	if (version == 0) {
		sql = "SELECT blob, version, timestamp FROM duckdb_ml_models WHERE model = ?1 ORDER BY version DESC LIMIT 1";
	} else {
		sql = "SELECT blob, version, timestamp FROM duckdb_ml_models WHERE model = ?1 AND version = ?2 LIMIT 1";
	}
	SqliteStatement stmt(db, sql);
	sqlite3_bind_text(stmt.Get(), 1, model_name.c_str(), -1, SQLITE_TRANSIENT);
	if (version > 0) {
		sqlite3_bind_int64(stmt.Get(), 2, UnsafeNumericCast<sqlite3_int64>(version));
	}

	auto rc = sqlite3_step(stmt.Get());
	if (rc == SQLITE_DONE) {
		if (version == 0) {
			throw InvalidInputException("No trained model named '%s' found in %s", model_name, connection.Path());
		}
		throw InvalidInputException("No trained model named '%s' with version %llu found in %s", model_name,
		                            static_cast<unsigned long long>(version), connection.Path());
	}
	if (rc != SQLITE_ROW) {
		throw IOException("Could not load model '%s' from '%s': %s", model_name, connection.Path(),
		                 sqlite3_errmsg(db));
	}

	auto *blob_data = static_cast<const char *>(sqlite3_column_blob(stmt.Get(), 0));
	auto blob_size = sqlite3_column_bytes(stmt.Get(), 0);
	auto found_version = UnsafeNumericCast<idx_t>(sqlite3_column_int64(stmt.Get(), 1));
	auto *timestamp_text = reinterpret_cast<const char *>(sqlite3_column_text(stmt.Get(), 2));

	RegistryLookupResult result;
	if (blob_data && blob_size > 0) {
		result.blob.assign(blob_data, UnsafeNumericCast<size_t>(blob_size));
	}
	result.version = found_version;
	if (timestamp_text) {
		result.timestamp = string(timestamp_text);
	}
	return result;
}

} // namespace ml
} // namespace duckdb
