//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//---------------------------------------------------------------------------
/// @file
///   This file includes APIs related to DSP on supported platforms
//---------------------------------------------------------------------------

#ifndef DSP_HPP
#define DSP_HPP

#include <mutex>
#include <string>

namespace pal {
class Dsp;
}

class pal::Dsp {
 public:
  //---------------------------------------------------------------------------
  /// @brief
  ///   This API is only for Windows platform.
  ///   Get the absolute location of DSP driver library (libcdsprpc.so/dll).
  /// @return
  ///   On success, return location of DSP driver library.
  ///   On error, return an empty string.
  //---------------------------------------------------------------------------
  static std::string getDspDriverPath();

 private:
  static std::mutex s_mutex;
};

#endif // DSP_HPP
