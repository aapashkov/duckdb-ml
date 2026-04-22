#define DUCKDB_EXTENSION_MAIN

#include "ml_extension.hpp"
#include "ml/register.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader *loader) {
	ml::RegisterMlFit(*loader);
	ml::RegisterMlPredict(*loader);
	ml::RegisterMlEvaluate(*loader);
	ml::RegisterMlExplain(*loader);
}

void MlExtension::Load(ExtensionLoader &loader) {
	LoadInternal(&loader);
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
	duckdb::LoadInternal(&loader);
}
}
