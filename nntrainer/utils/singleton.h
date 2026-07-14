// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   singleton.h
 * @date   11 July 2025
 * @brief  Base class making derived class singleton
 * @see    https://github.com/nntrainer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_SINGLETON_H__
#define __NNTRAINER_SINGLETON_H__

#include <mutex>

#include "noncopyable.h"
#include "nonmovable.h"

#if defined(__GNUC__) || defined(__clang__)
#define NNTR_VISIBILITY __attribute__((visibility("default")))
#else
#define NNTR_VISIBILITY
#endif

namespace nntrainer {

/**
 * @class   Singleton
 * @brief   Use it as base class of singletons
 */
template <typename T> class Singleton : public Noncopyable, public Nonmovable {

public:
  /**
   * @brief   Get reference to global instance object
   * once
   * @return Instance to singleton class reference
   */
  NNTR_VISIBILITY static T &Global() {
    static T instance;
    instance.initializeOnce();
    return instance;
  }

  /**
   * @brief   Initialize helper function ensuring that initialize is called only
   * once
   */
  void initializeOnce() {
    std::call_once(initialized_, [&]() { this->initialize(); });
  }

  /**
   * @brief   Default destructor
   */
  virtual ~Singleton() = default;

protected:
  /**
   * @brief   Default constructor
   */
  Singleton() = default;

  /**
   * @brief   Override this function to initialize derived class
   */
  virtual void initialize() noexcept {}

  std::once_flag initialized_;
};

} // namespace nntrainer

#endif // __NNTRAINER_SINGLETON_H__
