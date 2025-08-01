# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_EXTRA_INSTALL_TOOL_TARGETS)
set(_TRACY_ENABLED OFF)

if(IREE_BUILD_TRACY)
  message(STATUS "Bundling Tracy CLI tools with Python API")
  set(_TRACY_ENABLED ON)
  list(APPEND _EXTRA_INSTALL_TOOL_TARGETS "IREETracyCaptureServer")
endif()

################################################################################
# Package
################################################################################

# nanobind requires both RTTI and Exceptions, and it does not know that
# we have disabled them globally, so turn them back on. Since this is
# *the only* place in the codebase where we do this, just inline here.
# Note that this is playing with fire and the extension code is structured
# so as not to cause problems with RTTI cross-module issues.
iree_select_compiler_opts(_RTTI_AND_EXCEPTION_COPTS
  CLANG_OR_GCC
    "-frtti"
    "-fexceptions"
  MSVC_OR_CLANG_CL
    # Configure exception handling for standard C++ behavior.
    # - /EHs enables C++ catch-style exceptions
    # - /EHc breaks unwinding across extern C boundaries, dramatically reducing
    #   unwind table size and associated exception handling overhead as the
    #   compiler can assume no exception will ever be thrown within any function
    #   annotated with extern "C".
    # https://docs.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model
    "/EHsc"
    # Configure RTTI generation.
    # - /GR - Enable generation of RTTI (default)
    # - /GR- - Disables generation of RTTI
    # https://docs.microsoft.com/en-us/cpp/build/reference/gr-enable-run-time-type-information?view=msvc-160
    "/GR"
)

nanobind_add_module(iree_runtime_bindings_python_PyExtRt
  NB_STATIC LTO FREE_THREADED
  "binding.h"
  "initialize_module.cc"
  "invoke.h"
  "invoke.cc"
  "io.h"
  "io.cc"
  "hal.h"
  "hal.cc"
  "local_dlpack.h"
  "loop.h"
  "loop.cc"
  "numpy_interop.h"
  "numpy_interop.cc"
  "py_module.h"
  "py_module.cc"
  "status_utils.cc"
  "status_utils.h"
  "vm.h"
  "vm.cc"
)

target_link_libraries(iree_runtime_bindings_python_PyExtRt
  PRIVATE
  iree::base
  iree::base::internal::flags
  iree::hal
  iree::hal::drivers
  iree::hal::utils::allocators
  iree::base::internal::path
  iree::io::file_handle
  iree::io::formats::irpa
  iree::io::formats::parser_registry
  iree::io::parameter_index
  iree::io::parameter_index_provider
  iree::io::parameter_provider
  iree::io::scope_map
  iree::modules::io::parameters
  iree::modules::hal
  iree::schemas::parameter_archive
  iree::tooling::device_util
  iree::tooling::modules
  iree::vm
  iree::vm::bytecode::module

  Python::NumPy
)

target_compile_options(iree_runtime_bindings_python_PyExtRt
  PRIVATE
  ${IREE_DEFAULT_COPTS}
  # Default COPTS disable exceptions/rtti. Re-enable them.
  ${_RTTI_AND_EXCEPTION_COPTS}
)

set_target_properties(
  iree_runtime_bindings_python_PyExtRt
  PROPERTIES OUTPUT_NAME "iree/_runtime_libs/_runtime"
)

iree_py_library(
  NAME
    runtime
  SRCS
    "iree/runtime/__init__.py"
    "iree/runtime/_binding.py"
    "iree/runtime/_binding.pyi"
    "iree/runtime/array_interop.py"
    "iree/runtime/benchmark.py"
    "iree/runtime/flags.py"
    "iree/runtime/function.py"
    "iree/runtime/io.py"
    "iree/runtime/system_api.py"
    "iree/runtime/system_setup.py"
    "iree/runtime/typing.py"
    "iree/runtime/version.py"
    "iree/_runtime/__init__.py"
    "iree/_runtime/libs.py"
    "iree/_runtime/scripts/iree_benchmark_executable/__main__.py"
    "iree/_runtime/scripts/iree_benchmark_module/__main__.py"
    "iree/_runtime/scripts/iree_c_embed_data/__main__.py"
    "iree/_runtime/scripts/iree_cpuinfo/__main__.py"
    "iree/_runtime/scripts/iree_convert_parameters/__main__.py"
    "iree/_runtime/scripts/iree_create_parameters/__main__.py"
    "iree/_runtime/scripts/iree_dump_module/__main__.py"
    "iree/_runtime/scripts/iree_dump_parameters/__main__.py"
    "iree/_runtime/scripts/iree_flatcc_cli/__main__.py"
    "iree/_runtime/scripts/iree_run_module/__main__.py"
    "iree/_runtime/scripts/iree_tracy_capture/__main__.py"
  PYEXT_DEPS
    iree_runtime_bindings_python_PyExtRt
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-c-embed-data
  TO_EXE_NAME iree/_runtime_libs/iree-c-embed-data
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-benchmark-executable
  TO_EXE_NAME iree/_runtime_libs/iree-benchmark-executable
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-benchmark-module
  TO_EXE_NAME iree/_runtime_libs/iree-benchmark-module
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-convert-parameters
  TO_EXE_NAME iree/_runtime_libs/iree-convert-parameters
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-cpuinfo
  TO_EXE_NAME iree/_runtime_libs/iree-cpuinfo
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-create-parameters
  TO_EXE_NAME iree/_runtime_libs/iree-create-parameters
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-dump-module
  TO_EXE_NAME iree/_runtime_libs/iree-dump-module
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-dump-parameters
  TO_EXE_NAME iree/_runtime_libs/iree-dump-parameters
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-flatcc-cli
  TO_EXE_NAME iree/_runtime_libs/iree-flatcc-cli
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-run-module
  TO_EXE_NAME iree/_runtime_libs/iree-run-module
)

if(_TRACY_ENABLED)
  iree_symlink_tool(
    TARGET runtime
    FROM_TOOL_TARGET IREETracyCaptureServer
    TO_EXE_NAME iree/_runtime_libs/iree-tracy-capture
  )
endif()

################################################################################
# Tests
################################################################################

iree_py_test(
  NAME
    array_interop_test
  SRCS
    "tests/array_interop_test.py"
)

iree_py_test(
  NAME
    flags_test
  SRCS
    "tests/flags_test.py"
)

iree_py_test(
  NAME
    function_test
  SRCS
    "tests/function_test.py"
)

iree_py_test(
  NAME
    io_test
  SRCS
    "tests/io_test.py"
)

iree_py_test(
  NAME
    system_setup_test
  SRCS
    "tests/system_setup_test.py"
)

# These tests use compiler APIs as well as runtime APIs.
#
# These tests perform linking via the Compiler API, which is only supported
# in bundled-LLVM builds at the moment (#14086).
if(IREE_BUILD_COMPILER AND IREE_BUILD_BUNDLED_LLVM)
  iree_py_test(
    NAME
      benchmark_test
    SRCS
      "tests/benchmark_test.py"
  )

  iree_py_test(
    NAME
      hal_test
    SRCS
      "tests/hal_test.py"
  )

  iree_py_test(
    NAME
      io_runtime_test
    SRCS
      "tests/io_runtime_test.py"
  )

  iree_py_test(
    NAME
      system_api_test
    SRCS
      "tests/system_api_test.py"
  )

  iree_py_test(
    NAME
      vm_test
    SRCS
      "tests/vm_test.py"
  )
endif()

iree_py_test(
  NAME
    vm_types_test
  SRCS
    "tests/vm_types_test.py"
)

################################################################################
# Install
################################################################################

set(_INSTALL_DIR "python_packages/iree_runtime")
set(_INSTALL_COMPONENT "IreePythonPackage-runtime")

# Install iree/runtime/*.py files verbatim into the tree.
# We do this at the package level so as to avoid any loose files
# from outside (i.e. tests, etc).
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iree/runtime/"
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/iree/runtime/"
  FILES_MATCHING PATTERN "*.py"
)

# Install iree/runtime/*.py files verbatim into the tree.
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iree/_runtime/"
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/iree/_runtime/"
  FILES_MATCHING PATTERN "*.py"
)

# _runtime.so -> python_packages/iree_runtime/iree/_runtime.so
install(
  TARGETS iree_runtime_bindings_python_PyExtRt
  DESTINATION "${_INSTALL_DIR}/iree/_runtime_libs"
  COMPONENT "${_INSTALL_COMPONENT}"
)

# Install tools into python_packages/iree_runtime/iree/runtime
#
# Our runtime/... directory is included by the root CMakeLists before the
# tools/ directory which defines these targets, so we defer the install() to
# the end of the root file. While deferred calls are generally fragile, this
# install is purely a leaf feature (with no other calls depending on its
# sequencing), so this use is okay.
cmake_language(EVAL CODE "
cmake_language(DEFER DIRECTORY \"${IREE_SOURCE_DIR}\"
  CALL install
  TARGETS
    iree-cpuinfo
    iree-benchmark-executable
    iree-benchmark-module
    iree-c-embed-data
    iree-convert-parameters
    iree-create-parameters
    iree-dump-module
    iree-dump-parameters
    iree-flatcc-cli
    iree-run-module
    ${_EXTRA_INSTALL_TOOL_TARGETS}
  DESTINATION \"${_INSTALL_DIR}/iree/_runtime_libs\"
  COMPONENT \"${_INSTALL_COMPONENT}\"
)
")
