#ifndef __PMLC_DIALECT_LINALGX_PASSES__
#define __PMLC_DIALECT_LINALGX_PASSES__

include "mlir/Pass/PassBase.td"

def ReorderDimensions : FunctionPass<"linalgx-reorder-dimensions"> {
  let summary = "Reorder dimensions of complex arguments of lianlg.generic";
  let constructor = "pmlc::dialect::linalgx::createReorderDimensionsPass()";
}

#endif // __PMLC_DIALECT_LINALGX_PASSES__
