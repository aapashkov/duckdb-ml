// Copyright 2026
// Function registration declarations for ML extension SQL entry points.

#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace ml {

void RegisterMlFit(ExtensionLoader &loader);
void RegisterMlPredict(ExtensionLoader &loader);
void RegisterMlEvaluate(ExtensionLoader &loader);
void RegisterMlExplain(ExtensionLoader &loader);
void RegisterMlModels(ExtensionLoader &loader);

} // namespace ml
} // namespace duckdb
