 pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
  SRCS
    reorder_dims.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::pml::ir
    pmlc::util
    ::passes-gen
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
