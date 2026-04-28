// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/temp_file.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace iree::testing {

bool TempFilePath::Exists() const {
  if (path_.empty()) return false;
#if defined(IREE_PLATFORM_WINDOWS)
  return GetFileAttributesA(path_.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
  struct stat stat_buf;
  return stat(path_.c_str(), &stat_buf) == 0;
#endif  // IREE_PLATFORM_WINDOWS
}

bool TempFilePath::Remove() const {
  if (path_.empty()) return false;
#if defined(IREE_PLATFORM_WINDOWS)
  return DeleteFileA(path_.c_str()) != 0;
#else
  return unlink(path_.c_str()) == 0;
#endif  // IREE_PLATFORM_WINDOWS
}

}  // namespace iree::testing
