// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/Range.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_TESTFLOATRANGEANALYSISPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

class TestFloatRangeAnalysisPass
    : public impl::TestFloatRangeAnalysisPassBase<TestFloatRangeAnalysisPass> {
public:
  void runOnOperation() override {
    Explorer explorer(getOperation(), TraversalAction::SHALLOW);
    llvm::BumpPtrAllocator allocator;
    DFX::Solver solver(explorer, allocator);

    // Collect all probe points.
    SmallVector<std::pair<Operation *, const FloatRangeValueElement *>>
        queryOps;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "iree_unregistered.test_fprange" &&
          op->getNumOperands() == 1) {
        Value operand = op->getOperands().front();
        const FloatRangeValueElement &element =
            solver.getOrCreateElementFor<FloatRangeValueElement>(
                Position::forValue(operand));
        queryOps.emplace_back(op, &element);
      }
    });

    // Solve.
    if (failed(solver.run())) {
      return signalPassFailure();
    }

    // Update.
    for (auto &it : queryOps) {
      it.first->setAttr("analysis", StringAttr::get(&getContext(),
                                                    it.second->getAsStr(
                                                        solver.getAsmState())));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
