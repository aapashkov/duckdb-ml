#define DUCKDB_EXTENSION_MAIN

#include "ml_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

inline void MlScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Ml " + name.GetString() + " 🐥");
	});
}

inline void MlOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Ml " + name.GetString() + ", my linked OpenSSL version is " +
		                                           OPENSSL_VERSION_TEXT);
	});
}

static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto ml_scalar_function = ScalarFunction("ml", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlScalarFun);
	loader.RegisterFunction(ml_scalar_function);

	// Register another scalar function
	auto ml_openssl_version_scalar_function = ScalarFunction("ml_openssl_version", {LogicalType::VARCHAR},
	                                                            LogicalType::VARCHAR, MlOpenSSLVersionScalarFun);
	loader.RegisterFunction(ml_openssl_version_scalar_function);
}

void MlExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string MlExtension::Name() {
	return "ml";
}

std::string MlExtension::Version() const {
#ifdef EXT_VERSION_ML
	return EXT_VERSION_ML;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(ml, loader) {
	duckdb::LoadInternal(loader);
}
}
