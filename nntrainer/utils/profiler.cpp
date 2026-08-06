// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   profiler.cpp
 * @date   09 December 2020
 * @brief  Profiler related codes to be used to benchmark things
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>

namespace nntrainer {

/** @brief Return the Profiler singleton. */
template <>
NNTRAINER_SINGLETON_API profile::Profiler &
Singleton<profile::Profiler>::Global() {
  static profile::Profiler instance;
  instance.initializeOnce();
  return instance;
}

namespace profile {

void GenericProfileListener::onNotifyTimeEvent(
  PROFILE_EVENT event, const int time_item, const std::string &str,
  const std::chrono::microseconds &duration) {
  auto time_iter = time_taken.find(time_item);

  if (time_iter == time_taken.end()) {
    reset(time_item, str);
    time_iter = time_taken.find(time_item);
    if (time_iter == time_taken.end()) {
      throw std::runtime_error("Couldn't find time_iter.");
    }
  }

  auto &cnt_ = std::get<GenericProfileListener::CNT>(time_iter->second);
  cnt_++;

  if (warmups >= cnt_)
    return;

  auto &cur_ = std::get<GenericProfileListener::CUR>(time_iter->second);
  auto &min_ = std::get<GenericProfileListener::MIN>(time_iter->second);
  auto &max_ = std::get<GenericProfileListener::MAX>(time_iter->second);
  auto &sum_ = std::get<GenericProfileListener::SUM>(time_iter->second);

  cur_ = duration;
  min_ = std::min(min_, duration);
  max_ = std::max(max_, duration);
  sum_ += duration;
}

void GenericProfileListener::onNotifyMemoryEvent(
  PROFILE_EVENT event, const size_t alloc_current, const size_t alloc_total,
  const std::string &str, const std::chrono::microseconds &duration,
  const std::string &cache_policy, bool cache_fsu) {

  if (event != EVENT_MEM_ANNOTATE) {
    mem_max = std::max(mem_max, alloc_total);
    mem_sum += alloc_total;
    mem_average = mem_sum / ++mem_count;
  }

  mem_taken.emplace_back(event, alloc_current, alloc_total, str, duration,
                         cache_policy, cache_fsu);
}

void GenericProfileListener::notify(
  PROFILE_EVENT event, const std::shared_ptr<ProfileEventData> data) {
  switch (event) {
  case EVENT_TIME_START:
    /* ignore start time. we only consider duration of item */
    break;
  case EVENT_TIME_END:
    onNotifyTimeEvent(event, data->time_item, data->event_str, data->duration);
    break;
  case EVENT_MEM_ALLOC:
  case EVENT_MEM_DEALLOC:
  case EVENT_MEM_ANNOTATE:
    onNotifyMemoryEvent(event, data->alloc_current, data->alloc_total,
                        data->event_str, data->duration, data->cache_policy,
                        data->cache_fsu);
    break;
  default:
    throw std::runtime_error("Invalid PROFILE_EVENT");
    break;
  }
}

void GenericProfileListener::reset(const int time_item,
                                   const std::string &name) {
  time_taken[time_item] = std::make_tuple(
    std::chrono::microseconds{0}, std::chrono::microseconds::max(),
    std::chrono::microseconds::min(), std::chrono::microseconds{0}, int{0});
  names[time_item] = name;
}

const std::chrono::microseconds
GenericProfileListener::result(const int time_item) {
  auto iter = time_taken.find(time_item);

  if (iter == time_taken.end() ||
      std::get<GenericProfileListener::CNT>(iter->second) == 0) {
    std::stringstream ss;
    ss << "time_item has never recorded: " << names[time_item];
    throw std::invalid_argument("time_item has never recorded");
  }

  return std::get<GenericProfileListener::CUR>(iter->second);
}

void GenericProfileListener::report(std::ostream &out) const {
  std::vector<unsigned int> column_size = {20, 23, 23, 23, 23, 23, 23, 23};

  for (auto &[item, time] : time_taken) {
    auto title = names.find(item);
    if (title == names.end())
      throw std::runtime_error("Couldn't find name. it's already removed.");
    column_size[0] =
      std::max(column_size[0], static_cast<unsigned int>(title->second.size()));
  }
  auto total_col_size =
    std::accumulate(column_size.begin(), column_size.end(), 0);

  if (warmups != 0)
    out << "warm up: " << warmups << '\n';

  auto end = std::chrono::steady_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start_time);

  out << "profiled for " << duration.count() << '\n';

  /// creating header
  // clang-format off
  out << std::setw(column_size[0]) << "key"
      << std::setw(column_size[1]) << "avg"
      << std::setw(column_size[2]) << "min"
      << std::setw(column_size[3]) << "max"
      << std::setw(column_size[4]) << "sum"
      << std::setw(column_size[5]) << "pct" << '\n';
  // clang-format on

  // seperator
  out << std::string(total_col_size, '=') << '\n';

  std::map<int, std::function<void(std::ostream & out)>> ordered_report;

  /// calculate metrics while skipping warmups
  for (auto &time : time_taken) {
    auto func = [&](std::ostream &out_) {
      auto &cnt_ = std::get<GenericProfileListener::CNT>(time.second);
      auto &min_ = std::get<GenericProfileListener::MIN>(time.second);
      auto &max_ = std::get<GenericProfileListener::MAX>(time.second);
      auto &sum_ = std::get<GenericProfileListener::SUM>(time.second);

      auto title = names.find(time.first);
      if (title == names.end())
        throw std::runtime_error("Couldn't find name. it's already removed.");

      if (warmups >= cnt_) {
        out_ << std::left << std::setw(total_col_size) << title->second
             << "less data then warmup\n";
        out_
          << std::right; // Restore outputstream adjustflag to standard stream
        return;
      }

      out_.setf(std::ios::fixed);
      std::streamsize default_precision = out_.precision(2);
      // clang-format off
      out_ << std::setw(column_size[0]) << title->second
          << std::setw(column_size[1]) << sum_.count() / (cnt_ - warmups)
          << std::setw(column_size[2]) << min_.count()
          << std::setw(column_size[3]) << max_.count()
          << std::setw(column_size[4]) << sum_.count()
          << std::setw(column_size[5]) << sum_.count() / (double)duration.count() * 100 << '\n';
      // clang-format on
      out_.precision(default_precision);
      out_.unsetf(std::ios::fixed);
    };
    ordered_report[-time.first] = func;
  }

  for (auto &entry : ordered_report)
    entry.second(out);

  ordered_report.clear();

  column_size.clear();
  column_size = {30, 20, 20, 60, 10, 20, 10};

  out << std::string(total_col_size, '=') << '\n';
  out << "\n\n";
  /// creating header
  // clang-format off
  out << std::setw(column_size[0]) << "event"
      << std::setw(column_size[1]) << "size"
      << std::setw(column_size[2]) << "total"
      << std::setw(column_size[3]) << "info"
      << std::setw(column_size[4]) << "dur"
      << std::setw(column_size[5]) << "policy"
      << std::setw(column_size[6]) << "fsu"
      << std::endl;
  // clang-format on
  out << std::string(total_col_size, '=') << std::endl;

  int order = 0;
  for (auto &mem : mem_taken) {
    auto func = [&](std::ostream &out_) {
      auto &event = std::get<PROFILE_EVENT>(mem);
      auto &cur = std::get<1>(mem);
      auto &total = std::get<2>(mem);
      auto &info = std::get<3>(mem);
      auto &dur = std::get<std::chrono::microseconds>(mem);
      auto &policy = std::get<5>(mem);
      auto &fsu = std::get<bool>(mem);

      out_.setf(std::ios::fixed);
      out_.setf(std::ios::right);
      std::streamsize default_precision = out_.precision(2);
      // clang-format off
      if (event == EVENT_MEM_ANNOTATE) {
        out_ << info << std::endl;
      } else { 
        out_ << std::setw(column_size[0]) << ((event == EVENT_MEM_ALLOC) ? "ALLOC" :
                                              (event == EVENT_MEM_DEALLOC) ? "DEALLOC" : "")
             << std::setw(column_size[1]) << std::to_string(cur)
             << std::setw(column_size[2]) << std::to_string(total)
             << std::setw(column_size[3]) << info
             << std::setw(column_size[4]) << ((event == EVENT_MEM_DEALLOC) ? std::to_string(dur.count()) : "")
             << std::setw(column_size[5]) << policy
             << std::setw(column_size[6]) << (fsu ? ((event == EVENT_MEM_ALLOC) ? "IN" :
                                                     (event == EVENT_MEM_DEALLOC) ? "OUT" : "") : "")
             << std::endl;
      }
      // clang-format on
      out_.precision(default_precision);
      out_.unsetf(std::ios::fixed);
    };
    ordered_report[order++] = func;
  }

  for (auto &entry : ordered_report)
    entry.second(out);

  out << "Max Memory Size = " << mem_max << std::endl;
  out << "Average Memory Size = " << mem_average << std::endl;
}

void Profiler::start(const int item) {
#ifdef DEBUG
  /// @todo: consider race condition
  if (time_item_times.find(item) != time_item_times.end())
    throw std::invalid_argument("profiler has already started");
#endif

  auto name = time_item_names.find(item);
  if (name == time_item_names.end())
    throw std::invalid_argument("the item is not registered");

  time_item_times[item] = std::chrono::steady_clock::now();

  auto data = std::make_shared<ProfileEventData>(item, 0, 0, name->second,
                                                 std::chrono::microseconds(0));
  notifyListeners(EVENT_TIME_START, data);
}

void Profiler::end(const int item) {
  /// @todo: consider race condition
  auto end = std::chrono::steady_clock::now();

#ifdef DEBUG
  if (time_item_times.find(item) == time_item_times.end())
    throw std::invalid_argument("profiler hasn't started with the item");
#endif

  auto name = time_item_names.find(item);
  if (name == time_item_names.end())
    throw std::invalid_argument("the item is not registered");

  auto start = time_item_times[item];
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto data =
    std::make_shared<ProfileEventData>(item, 0, 0, name->second, duration);

  notifyListeners(EVENT_TIME_END, data);

  time_item_times.erase(item);
}

void Profiler::notifyListeners(PROFILE_EVENT event,
                               const std::shared_ptr<ProfileEventData> data) {
  std::lock_guard<std::mutex> lock(listeners_mutex);

  for (auto &l : time_item_listeners[data->time_item])
    l->notify(event, data);

  for (auto &l : listeners)
    l->notify(event, data);
}

void Profiler::subscribe(std::shared_ptr<ProfileListener> listener,
                         const std::set<int> &time_items) {
  if (listener == nullptr) {
    throw std::invalid_argument("listener is null!");
  }

  std::lock_guard<std::mutex> lock(listeners_mutex);
  if (time_items.empty()) {
    listeners.insert(listener);
  } else {
    for (auto item : time_items)
      time_item_listeners[item].insert(listener);
  }
}

void Profiler::unsubscribe(std::shared_ptr<ProfileListener> listener) {
  std::lock_guard<std::mutex> lock(listeners_mutex);
  listeners.erase(listener);

  for (auto &[item, listeners] : time_item_listeners) {
    auto found = listeners.find(listener);
    if (found != listeners.end())
      listeners.erase(found);
  }
}

int Profiler::registerTimeItem(const std::string &name) {
  std::lock_guard<std::mutex> lock_listener(listeners_mutex);
  std::lock_guard<std::mutex> lock(registr_mutex);

  int item = time_item_names.size() + 1;

  time_item_names[item] = name;
  time_item_listeners[item] = std::set<std::shared_ptr<ProfileListener>>();

  ml_logd("[Profiler] Event registered, time item name: %s key: %d",
          name.c_str(), item);

  return item;
}

void Profiler::alloc(const void *ptr, size_t size, const std::string &str,
                     const std::string &policy, bool fsu) {
  std::lock_guard<std::mutex> lock(allocates_mutex);

#ifdef DEBUG
  auto found = allocates.find(ptr);
  if (found != allocates.end())
    throw std::invalid_argument("memory profiler is already allocated");
#endif

  allocates[ptr] = std::tuple<size_t, timepoint, std::string>(
    size, std::chrono::steady_clock::now(), str);

  total_size += size;

  auto data = std::make_shared<ProfileEventData>(
    0, size, total_size.load(), str, std::chrono::microseconds(0), policy, fsu);
  notifyListeners(EVENT_MEM_ALLOC, data);
}

void Profiler::dealloc(const void *ptr, const std::string &policy, bool fsu) {
  std::lock_guard<std::mutex> lock(allocates_mutex);

  auto end = std::chrono::steady_clock::now();
  auto found = allocates.find(ptr);

  if (found == allocates.end())
    throw std::invalid_argument("memory profiler didn't allocated");

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    end - std::get<timepoint>(found->second));

  auto size = std::get<size_t>(found->second);
  total_size -= size;

  auto str = std::get<std::string>(found->second);
  auto data = std::make_shared<ProfileEventData>(0, size, total_size.load(),
                                                 str, duration, policy, fsu);

  notifyListeners(EVENT_MEM_DEALLOC, data);

  allocates.erase(found);
}

void Profiler::annotate(const std::string &str) {
  std::lock_guard<std::mutex> lock(allocates_mutex);

  auto data = std::make_shared<ProfileEventData>(0, 0, 0, str,
                                                 std::chrono::microseconds(0));

  notifyListeners(EVENT_MEM_ANNOTATE, data);
}

} // namespace profile

} // namespace nntrainer
