// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   graph_node.h
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the graph node interface for c++ API
 */

#ifndef __GRAPH_NODE_H__
#define __GRAPH_NODE_H__

#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace nntrainer {

/**
 * @class   Layer Base class for the graph node
 * @brief   Base class for all layers
 */
class GraphNode {
public:
  /**
   * @brief Provides the time/order at which the node will be executed.
   * @details This time will be finalized once the graph has been calculated.
   * Each element indicates the orders with which the below operations
   * for each node are executed:
   * 1. Forwarding
   * 2. calcGradient
   * 3. calcDerivative
   * 4. ApplyGradient
   * One constraint is that they must be sorted in ascending order.
   * This ensures that the operations are executed in the order of their
   * listing.
   */
  typedef std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>
    ExecutionOrder;

  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~GraphNode() = default;

  /**
   * @brief     Get the Name of the underlying object
   *
   * @return std::string Name of the underlying object
   * @note name of each node in the graph must be unique
   */
  virtual const std::string getName() const = 0;

  /**
   * @brief     Set the Name of the underlying object
   *
   * @param[in] std::string Name for the underlying object
   * @note name of each node in the graph must be unique, and caller must ensure
   * that
   */
  virtual void setName(const std::string &name) = 0;

  /**
   * @brief     Get the Type of the underlying object
   *
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

  /**
   * @brief     Get the trainable parameter
   *
   * @return bool true / false
   */
  virtual bool getTrainable() const = 0;

  /**
   * @brief     Get the input connections for this node
   *
   * @return list of name of the nodes which form input connections
   */
  virtual const std::vector<std::string> getInputConnections() const = 0;

  /**
   * @brief     Get the output connections for this node
   *
   * @return list of name of the nodes which form output connections
   */
  virtual const std::vector<std::string> getOutputConnections() const = 0;

  /**
   * @brief     get the execution order/location of this node
   *
   * @retval    the execution order/location of this node
   * @details   The two values represents the value for forward and backward
   * respectively
   */
  virtual ExecutionOrder getExecutionOrder() const = 0;

  /**
   * @brief     set the execution order/location of this node
   *
   * @param     exec_order the execution order/location of this node
   * @details   The two values represents the value for forward and backward
   * respectively
   */
  virtual void setExecutionOrder(ExecutionOrder exec_order_) = 0;

  /**
   * @brief Check if this node can support in-place execution
   * @return true if inplace, else false
   * @note Used by memory manager to determine input tensor lifespan
   */
  virtual bool supportInPlace() const { return false; }
};

/**
 * @brief   Iterator for GraphNode which return const
 * std::shared_ptr<LayerNodeType> object upon realize
 *
 * @note    This does not include the complete list of required functions. Add
 * them as per need.
 *
 * @note    GraphNodeType is to enable for both GraphNode and const GraphNode
 */
template <typename LayerNodeType, typename GraphNodeType>
class GraphNodeIterator {
  GraphNodeType *p; /** underlying object of GraphNode */

public:
  /**
   * @brief   iterator_traits types definition
   *
   * @note    these are not required to be explicitly defined now, but maintains
   *          forward compatibility for c++17 and later
   *
   * @note    value_type, pointer and reference are different from standard
   * iterator
   */
  typedef const std::shared_ptr<LayerNodeType> value_type;
  typedef std::random_access_iterator_tag iterator_category;
  typedef std::ptrdiff_t difference_type;
  typedef const std::shared_ptr<LayerNodeType> *pointer;
  typedef const std::shared_ptr<LayerNodeType> &reference;

  /**
   * @brief Construct a new Graph Node Iterator object
   *
   * @param x underlying object of GraphNode
   */
  GraphNodeIterator(GraphNodeType *x) : p(x) {}

  /**
   * @brief reference operator
   *
   * @return value_type
   * @note this is different from standard iterator
   */
  value_type operator*() const {
    return std::static_pointer_cast<LayerNodeType>(*p);
  }

  /**
   * @brief pointer operator
   *
   * @return value_type
   * @note this is different from standard iterator
   */
  value_type operator->() const {
    return std::static_pointer_cast<LayerNodeType>(*p);
  }

  /**
   * @brief == comparison operator override
   *
   * @param lhs iterator lhs
   * @param rhs iterator rhs
   * @retval true if match
   * @retval false if mismatch
   */
  friend bool operator==(GraphNodeIterator const &lhs,
                         GraphNodeIterator const &rhs) {
    return lhs.p == rhs.p;
  }

  /**
   * @brief != comparison operator override
   *
   * @param lhs iterator lhs
   * @param rhs iterator rhs
   * @retval true if mismatch
   * @retval false if match
   */
  friend bool operator!=(GraphNodeIterator const &lhs,
                         GraphNodeIterator const &rhs) {
    return lhs.p != rhs.p;
  }

  /**
   * @brief <= comparison operator override
   *
   * @param lhs iterator lhs
   * @param rhs iterator rhs
   * @retval true if left is less than or equal to the right value
   * @retval false if left is greater than the right value
   */
  friend bool operator<=(GraphNodeIterator const &lhs,
                         GraphNodeIterator const &rhs) {
    return lhs.p <= rhs.p;
  }

  /**
   * @brief override for ++ operator
   *
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator++() {
    p += 1;
    return *this;
  }

  /**
   * @brief override for operator++
   *
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator++(int) {
    GraphNodeIterator temp(p);
    p += 1;
    return temp;
  }

  /**
   * @brief override for -- operator
   *
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator--() {
    p -= 1;
    return *this;
  }

  /**
   * @brief override for operator--
   *
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator--(int) {
    GraphNodeIterator temp(p);
    p -= 1;
    return temp;
  }

  /**
   * @brief override for subtract operator
   *
   * @param offset offset to subtract
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator-(const difference_type offset) const {
    return GraphNodeIterator(p - offset);
  }

  /**
   * @brief override for subtract operator
   *
   * @param other iterator to subtract
   * @return difference_type
   */
  difference_type operator-(const GraphNodeIterator &other) const {
    return p - other.p;
  }

  /**
   * @brief override for subtract and return result operator
   *
   * @param offset offset to subtract
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator-=(const difference_type offset) {
    p -= offset;
    return *this;
  }

  /**
   * @brief override for add operator
   *
   * @param offset offset to add
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator+(const difference_type offset) const {
    return GraphNodeIterator(p + offset);
  }

  /**
   * @brief override for add and return result operator
   *
   * @param offset offset to add
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator+=(const difference_type offset) {
    p += offset;
    return *this;
  }
};

/**
 * @brief   Reverse Iterator for GraphNode which return LayerNode object upon
 * realize
 *
 * @note    This just extends GraphNodeIterator and is limited by its
 * functionality.
 */
template <typename T_iterator>
class GraphNodeReverseIterator : public std::reverse_iterator<T_iterator> {
public:
  /**
   * @brief Construct a new Graph Node Reverse Iterator object
   *
   * @param iter Iterator
   */
  explicit GraphNodeReverseIterator(T_iterator iter) :
    std::reverse_iterator<T_iterator>(iter) {}

  /**
   *  @brief reference operator
   *
   * @return T_iterator::value_type
   * @note this is different from standard iterator
   */
  typename T_iterator::value_type operator*() const {
    auto temp = std::reverse_iterator<T_iterator>::current - 1;
    return *temp;
  }

  /**
   *  @brief pointer operator
   *
   * @return T_iterator::value_type
   * @note this is different from standard iterator
   */
  typename T_iterator::value_type operator->() const {
    auto temp = std::reverse_iterator<T_iterator>::current - 1;
    return *temp;
  }
};

/**
 * @brief     Iterators to traverse the graph
 */
template <class LayerNodeType>
using graph_const_iterator =
  GraphNodeIterator<LayerNodeType, const std::shared_ptr<GraphNode>>;

/**
 * @brief     Iterators to traverse the graph
 */
template <class LayerNodeType>
using graph_const_reverse_iterator = GraphNodeReverseIterator<
  GraphNodeIterator<LayerNodeType, const std::shared_ptr<GraphNode>>>;

} // namespace nntrainer
#endif // __GRAPH_NODE_H__
