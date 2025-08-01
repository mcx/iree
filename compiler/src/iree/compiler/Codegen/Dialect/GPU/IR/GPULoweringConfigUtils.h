// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Helper to retrieve/set a target mma intrinsic.
IREE::Codegen::InnerTileDescAttrInterface getMmaKind(LoweringConfigAttr config);
void setMmaKind(MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
                IREE::Codegen::InnerTileDescAttrInterface kind);

// TODO: Merge subgroup counts functionality into subgroup tiling level
//       lowering, when we have it implemented.
/// Helper to retrieve/set a target subgroup M/N counts.
std::optional<int64_t> getSubgroupMCount(LoweringConfigAttr config);
std::optional<int64_t> getSubgroupNCount(LoweringConfigAttr config);
void setSubgroupMCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroupMCount);
void setSubgroupNCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroupNCount);

// The basis consists of two integer arrays:
//   - "counts": number of resource to use per dimension in the basis.
//   - "mapping": a projected permutation to map to basis to the operations
//     iteration space.
//
// Given a resource "x", the "basis" can be used to determine the distribution
// of an iteration space using:
//
// b = delinearize(x, counts)
// idx = apply(b, mapping)
struct Basis {
  SmallVector<int64_t> counts;
  SmallVector<int64_t> mapping;
};

// Helper to retrieve/set distribution basis.
FailureOr<Basis> getBasis(IREE::GPU::LoweringConfigAttr config,
                          IREE::GPU::TilingLevel level);
void setBasis(MLIRContext *context, SmallVector<NamedAttribute> &attrs,
              IREE::GPU::TilingLevel level, const Basis &basis);

/// Helper to retrieve a list of operand indices to promote.
std::optional<SmallVector<int64_t>>
getPromotedOperandList(LoweringConfigAttr config);
/// Helper to retrieve a list of operand promotion types.
std::optional<ArrayRef<Attribute>>
getPromotionTypesList(LoweringConfigAttr config);
/// Append to `attrs` an `ArrayAttr` for `promotedOperands`.
/// The `promotionTypes` is an optional list of Attributes
/// describing how to promote each individual operand.
void appendPromotedOperandsList(MLIRContext *context,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                ArrayRef<int64_t> operands,
                                ArrayRef<Attribute> promotionTypes = {});
/// Create a new `LoweringConfigAttr` from `currAttr` with the promoted operands
/// list modified/set to `operands`. Optional `promotionTypes` specifies how to
/// promote each operand.
IREE::GPU::LoweringConfigAttr setPromotedOperandsList(
    MLIRContext *context, IREE::GPU::LoweringConfigAttr currAttr,
    ArrayRef<int64_t> operands,
    std::optional<ArrayRef<Attribute>> promotionTypes = std::nullopt);

/// Helper to retrieve list of operand to pad.
std::optional<SmallVector<int64_t>> getPaddingList(LoweringConfigAttr config,
                                                   bool paddingConv = false);

IREE::GPU::UKernelConfigAttr
getUkernelSpec(IREE::GPU::LoweringConfigAttr config);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
