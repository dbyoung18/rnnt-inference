#include <ATen/core/grad_mode.h>
#include <c10/util/TypeCast.h>
#include <vector>
#include <list>
#include <chrono>
#include <algorithm>
#include <condition_variable>
#include <type_traits>
#include <loadgen.h>
#include <query_sample.h>
#include <iostream>
#include <fstream>

#include "torch_sut.hpp"

using Stack = std::vector<at::IValue>;

ProfileRecord::ProfileRecord(
  bool is_record,
  const std::string& profiler_file
) : is_record_(is_record), profiler_file_(profiler_file) {
  if (is_record_)
  {
    torch_profiler = std::make_unique<torch::autograd::profiler::RecordProfile>(profiler_file_);
  }
}

RNNTSUT::RNNTSUT(
    const std::string& model_file,
    const std::string& sample_file,
    const std::string& preprocessor_file,
    int inter_parallel,
    int intra_parallel,
    int batch, bool ht,
    bool profiler,
    const std::string& profiler_folder,
    bool preprocessor,
    std::string test_scenario,
    int perf_count
  ) : qsl_(sample_file), model_(model_file),
  preprocessor_(preprocessor_file), mThreshold_(batch),
  nProcsPerInstance_(intra_parallel), nInstances_(inter_parallel), mHt_(ht),
  profiler_flag_(profiler), profiler_folder_(profiler_folder),
  preprocessor_flag_(preprocessor), test_scenario_(test_scenario), perf_count_(perf_count) {

  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  if (nProcsPerInstance_ * nInstances_ > nMaxProc)
    nInstances_ = nMaxProc / nProcsPerInstance_;

  // Construct instances
  for (int i = 0; i < nInstances_; ++ i)
    mInstances_.emplace_back(&RNNTSUT::thInstance, this, i);
}

//
// TODO: Use hierachy information to allocate place
//
int RNNTSUT::rootProc(int index) {
  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  // XXX : Assumed 2-sockets, HT on !!!
  int part[] = {nMaxProc, nMaxProc*(2 + (int)mHt_)/4};

  auto select = index & 1;
  auto root = part[select] - nProcsPerInstance_ * ((index>>1) + 1);

  // Assert root > 0
  return root;
}

void RNNTSUT::thInstance(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nProcsPerInstance_);
  auto root = rootProc(index);
  auto which = index & 1;

  for (int i = 0; i < nProcsPerInstance_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  Queue_t snippet;

  // Wait for work
  std::string log_name;
  if (profiler_flag_) {
    log_name = profiler_folder_ + "/" + Name() + "_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    std::ofstream out(log_name);
  }
  {
    auto guard_ = std::make_unique<ProfileRecord>(profiler_flag_, log_name);
    size_t nIteration = 0;
    while (true) {
      // critical section
      {
        std::unique_lock<std::mutex> l(mtx_);
        ctrl_.wait(l, [this] {return mStop_ || !mQueue_.empty();});

        if (mStop_)
          break;

        auto nPeek = std::min(mQueue_.size(), mThreshold_);
        auto it = mQueue_.begin();
        // XXX: pointer chaser, choose better solution
        std::advance(it, nPeek);
        snippet.clear();
        snippet.splice(snippet.begin(), mQueue_, mQueue_.begin(), it);
        // if (!mQueue_.empty()) Offline never empty
        ctrl_.notify_one();
      }

      std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());
      std::vector<mlperf::QuerySampleIndex> indices (samples.size());
      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
          [](mlperf::QuerySample sample) {return sample.index;});
      Stack fea_stack;
      if (preprocessor_flag_) {
        auto wav_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
        auto pre_results = preprocessor_.inference_at(which, wav_stack);
        fea_stack = qsl_.GetIValueListFrom(pre_results);
        //QuerySamplesComplete(samples, at::stack(tList[0], -1));
      } else {
        fea_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
      }
      auto results = model_.inference_at(which, fea_stack);
      QuerySamplesComplete(samples, results);

      nIteration += 1;
      if (nIteration * mThreshold_ >= perf_count_)
        guard_->~ProfileRecord();
    }
  }
}

void RNNTSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  if (test_scenario_ == "Offline") {
    // Parallel sort samples into a queue
    mQueue_ = qsl_.Sort(samples, preprocessor_flag_);
  } else {
    for (auto sample : samples)
      mQueue_.emplace_back(sample);
  }
  ctrl_.notify_one();
}

void RNNTSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const std::vector<std::vector<int64_t>>& results) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    std::cout << samples[i].index << "::" << SequenceToString(results[i]) << std::endl;
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(results[i].data());
    responses[i].size = results[i].size()*sizeof(int64_t);
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

void RNNTSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const at::Tensor& results) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(results[i].data_ptr());
    responses[i].size = results[i].nbytes();
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

std::string RNNTSUT::SequenceToString(const std::vector<int64_t>& seq) {
  std::string str = "";
  std::vector<char> labels = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                              't', 'u', 'v', 'w', 'x', 'y', 'z', '\''};
  for (auto ch : seq)
    str.push_back(labels[ch]);
  return str;	
}

RNNTSUT::~RNNTSUT() {
  {
    std::unique_lock<std::mutex> l(mtx_);
    mStop_ = true;
    ctrl_.notify_all();
  }

  for (auto& Instance : mInstances_)
    Instance.join();
}

