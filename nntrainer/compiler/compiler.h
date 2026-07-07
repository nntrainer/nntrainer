// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler.h
 * @date 01 April 2021
 * @brief NNTrainer compiler that reads to generate optimized graph
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 * @details
 * Graph is convertible either to iostream, representation, executable by
 * appropriate compiler and interpreter
 * For example, if istream would be from a a.tflite file,
 *
 * GraphRepresentation g;
 * GraphCompiler * compiler = new NNTrainerCPUCompiler;
 *
 * ExecutableGraph eg = compiler->compile(g);
 *
 *
 *    +-------+--+--------+
 *    |GraphRepresentation|
 *    +-------+-----------+
 *            |  ^
 *  compile() |  |
 *            |  |
 *         (Compiler)
 *            |  |
 *            |  | decompile()
 *            v  |
 *      +--------+------+
 *      |ExecutableGraph|
 *      +---------------+
 *
 */
#ifndef __COMPILER_H__
#define __COMPILER_H__

#include <memory>

#include <compiler_fwd.h>
#include <network_graph.h>

namespace nntrainer {

/**
 * @brief Pure virtual class for the Graph Compiler
 *
 */
class GraphCompiler {
public:
  virtual ~GraphCompiler() {}
  /**
   * @brief Compile a graph representation into an executable graph.
   *
   * @param representation graph representation to compile
   * @return executable graph
   */
  virtual std::shared_ptr<ExecutableGraph>
  compile(std::shared_ptr<const GraphRepresentation> representation) = 0;

  /**
   * @brief Decompile an executable graph into a graph representation.
   *
   * @param executable executable graph to decompile
   * @return graph representation
   */
  virtual std::shared_ptr<GraphRepresentation>
  decompile(std::shared_ptr<ExecutableGraph> executable) = 0;
};

} // namespace nntrainer

#endif // __COMPILER_H__
