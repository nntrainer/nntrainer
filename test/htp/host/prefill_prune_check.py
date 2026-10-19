#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production prefetch join/cancellation with controlled readers."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sanitize', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
text = (root/'nntrainer/tensor/htp_backend/htp_compute_ops.cpp').read_text()
start = text.index('  std::vector<const void *> prefetch_qs4cx_wh_experts_end(')
end = text.index('\n  void reserve_qs4cx_wh_expert_slots(', start)
method = text[start:end].replace(' override {', ' {')
source = r'''
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
using remote_handle64 = unsigned long;
struct HtpBackend {
  static HtpBackend &global() { static HtpBackend b; return b; }
  unsigned handle() { return 0; }
};
struct HtpProfile {
  static uint64_t nowUs() { return 1; }
  static HtpProfile &global() { static HtpProfile p; return p; }
  int level() { return 0; }
  template <typename... T> void addPrefetch(T...) {}
};
struct Timing { uint64_t io_ns=0, copy_ns=0, calls=0, requested_bytes=0; };
struct Trace { uint64_t start_us=0, end_us=0; Timing timing; unsigned cpu=0; };
struct StagedExpert { struct { const void *key_gu; uint64_t off_gu; } d; int slot, rc=0; };
struct PrefetchBatch {
  std::vector<StagedExpert> experts;
  std::vector<bool> cancelled;
  size_t done=0;
  uint64_t trace_id=0, queued_us=0;
  size_t jobs_ahead=0;
  std::vector<Trace> read_trace;
};
struct Readers { unsigned load() { return 0; } };
struct Backend {
  std::mutex prefetch_mutex_, handle_mutex_;
  std::condition_variable prefetch_done_cv_;
  std::deque<std::unique_ptr<PrefetchBatch>> prefetch_batches_;
  std::deque<std::pair<PrefetchBatch *,size_t>> prefetch_jobs_;
  std::vector<int> free_expert_slots_;
  std::set<const void *> experts_;
  unsigned prefetch_caller_cpus_=0;
  Readers prefetch_reader_cpus_;
  void registerStagedBatch(remote_handle64, std::vector<StagedExpert> &live) {
    bool failed=false;
    for (const auto &st:live) {
      if(st.rc) { free_expert_slots_.push_back(st.slot); failed=true; }
      else experts_.insert(st.d.key_gu);
    }
    if(failed) throw std::runtime_error("read failed");
  }
METHOD
};
int keys[8];
PrefetchBatch *queue(Backend &b,int first,int count) {
  auto owned=std::make_unique<PrefetchBatch>(); auto *p=owned.get();
  for(int i=first;i<first+count;++i) p->experts.push_back({{&keys[i],0},i});
  p->cancelled.resize(count,false);
  for(int i=0;i<count;++i) b.prefetch_jobs_.emplace_back(p,i);
  b.prefetch_batches_.push_back(std::move(owned));return p;
}
int main() {
  // A claimed reader and an unread required expert must both finish. An
  // unread unused expert is cancelled; completed unused weights are retained.
  Backend b; auto *a=queue(b,0,4);auto *later=queue(b,4,2);
  b.prefetch_jobs_.pop_front(); // reader owns expert 0
  b.prefetch_jobs_.erase(b.prefetch_jobs_.begin()+2); // expert 3 finished
  a->done=1;
  std::vector<const void*> needed{&keys[0],&keys[1]}, cancelled;
  auto result=std::async(std::launch::async,[&]{return b.prefetch_qs4cx_wh_experts_end(&needed,&cancelled);});
  // Wait for the join to remove the unused job while holding the same mutex
  // used by the real reader to claim work. The deadline prevents test hangs.
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  for(;;) {
    { std::lock_guard<std::mutex> lock(b.prefetch_mutex_);
      if(a->cancelled[2]) break;
    }
    assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();
  }
  assert(result.wait_for(std::chrono::milliseconds(1))==std::future_status::timeout);
  assert(b.free_expert_slots_.empty());
  {
    std::lock_guard<std::mutex> lock(b.prefetch_mutex_);
    assert(a->done==2); assert(b.prefetch_jobs_.front().second==1);
    b.prefetch_jobs_.pop_front(); // reader takes required expert 1
    a->done+=2; // both claimed readers finish before the caller can recycle
  }
  b.prefetch_done_cv_.notify_all();
  assert(result.get()==std::vector<const void*>({&keys[0],&keys[1],&keys[3]}));
  assert(cancelled==std::vector<const void*>({&keys[2]}));
  assert(b.free_expert_slots_==std::vector<int>({2}));
  assert(b.prefetch_jobs_.size()==2 && b.prefetch_jobs_.front().first==later);
  // Fully cancelled batch: no read, no registration, every reservation freed.
  needed.clear(); cancelled.clear();
  assert(b.prefetch_qs4cx_wh_experts_end(&needed,&cancelled).empty());
  assert(cancelled.size()==2 && b.prefetch_jobs_.empty());
  assert(b.free_expert_slots_==std::vector<int>({2,4,5}));
  // Ordinary drain retains all reads and still propagates genuine I/O errors.
  a=queue(b,6,2);b.prefetch_jobs_.clear();a->done=2;a->experts[1].rc=EIO;
  bool threw=false;try{b.prefetch_qs4cx_wh_experts_end();}catch(const std::runtime_error&){threw=true;}
  assert(threw && b.experts_.count(&keys[6]) && !b.experts_.count(&keys[7]));
  assert(b.free_expert_slots_.back()==7);
  assert(b.prefetch_qs4cx_wh_experts_end().empty());
  puts("prefill prune: required/claimed reads, cancellation, FIFO, slots and errors PASS");
}
'''.replace('METHOD', method)
with tempfile.TemporaryDirectory(prefix='nntr-prefill-prune-') as tmp:
    path=Path(tmp)/'check.cpp';path.write_text(source)
    cmd=['g++','-std=c++17','-pthread','-O1','-g',str(path),'-o',str(Path(tmp)/'check')]
    if args.sanitize:
        cmd += ['-fsanitize=address,undefined','-fno-omit-frame-pointer','-no-pie']
    subprocess.run(cmd,check=True)
    subprocess.run([str(Path(tmp)/'check')],check=True,timeout=15)
