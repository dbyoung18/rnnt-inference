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

static const int preprocess_split = 1;
static const int preprocess_start_idx = 1;
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
    const std::string& sample_file,
    const std::string& model_file,
    const std::string& preprocessor_file,
    int inter_parallel,
    int intra_parallel,
    int batch_size, int split_len, bool enable_bf16, bool ht, bool pipeline,
    bool preprocessor,
    std::string test_scenario,
    bool profiler,
    const std::string& profiler_folder,
    int profiler_iter
  ) : qsl_(sample_file), model_(model_file, split_len, enable_bf16), preprocessor_(preprocessor_file), mQueuePreprocessed_(3000),
  nInstances_(inter_parallel), nProcsPerInstance_(intra_parallel),
  mThreshold_(batch_size), mHt_(ht), pipeline_flag_(pipeline), preprocessor_flag_(preprocessor),
  test_scenario_(test_scenario), profiler_flag_(profiler),
  profiler_folder_(profiler_folder), profiler_iter_(profiler_iter) {

  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  if (nProcsPerInstance_ * nInstances_ > nMaxProc)
    nInstances_ = nMaxProc / nProcsPerInstance_;

  // Construct instances
  if (pipeline_flag_ && preprocessor_flag_) {
    std::cout << "Use pipeline mode!" << std::endl;
    for (int i = 0; i < preprocess_start_idx; ++ i)
      mInstances_.emplace_back(&RNNTSUT::thInstancePreprocess, this, i);
    for (int i = preprocess_start_idx; i < nInstances_; ++ i)
      mInstances_.emplace_back(&RNNTSUT::thInstanceModel, this, i);
  } else {
    for (int i = 0; i < nInstances_; ++ i)
      mInstances_.emplace_back(&RNNTSUT::thInstance, this, i);
  }

  batch_sort_ = (test_scenario == "Offline");
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

void RNNTSUT::thInstancePreprocess(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nProcsPerInstance_);
  auto root = rootProc(index);
  auto which = index & 1;

  for (int i = 0; i < nProcsPerInstance_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  Queue_t snippet;

  {
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
        //std::cout << "splice mQueue remain " << mQueue_.size() << " index " << index << std::endl;
        std::cout << "Preprocessing index " << index << " size " << snippet.size() << std::endl;
        // if (!mQueue_.empty()) Offline never empty
        ctrl_.notify_one();
      }

      std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());

      qsl::Stack fea_stack;
      if (preprocess_split == 1) {
      std::vector<mlperf::QuerySampleIndex> indices(samples.size());
      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
        [](mlperf::QuerySample sample) {return sample.index;});
        auto wav_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
        auto pre_results = preprocessor_.inference_at(which, wav_stack);
        fea_stack = qsl_.GetIValueListFrom(pre_results);
        // Test preprocessor only(response {N, C, T})
        // QuerySamplesComplete(samples, fea_stack[0].toTensor());
        // continue;
        // {N, C, T} -> {T, N, C}
        fea_stack = {fea_stack[0].toTensor().permute({2, 0, 1}).contiguous(), fea_stack[1]};
      mQueuePreprocessed_.enqueue({fea_stack, samples});
      } else {
      at::Tensor x[preprocess_split];
      at::Tensor x_lens[preprocess_split];
      for (auto i = 0; i < preprocess_split; i ++) {
        std::vector<mlperf::QuerySampleIndex> indices(mThreshold_ / preprocess_split);
        std::transform(samples.cbegin() + i * indices.capacity(), samples.cbegin() + (i+1) * indices.capacity(), indices.begin(),
          [](mlperf::QuerySample sample) {return sample.index;});
        auto wav_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
        auto pre_results = preprocessor_.inference_at(which, wav_stack);
        fea_stack = qsl_.GetIValueListFrom(pre_results);
        // Test preprocessor only(response {N, C, T})
        // QuerySamplesComplete(samples, fea_stack[0].toTensor());
        // continue;
        // {N, C, T} -> {T, N, C}
        x[i] = fea_stack[0].toTensor().permute({2, 0, 1});
        x_lens[i] = fea_stack[1].toTensor();
      }
      auto xs = at::concat(x, 1).contiguous();
      auto xs_lens = at::concat(x_lens).contiguous();
      qsl::Stack concat_fea = {xs, xs_lens};
      mQueuePreprocessed_.enqueue({concat_fea, samples});
      }
    }
  }
}

void RNNTSUT::thInstanceModel(int index) {
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
  //std::string log_name;
  //if (profiler_flag_) {
    //log_name = profiler_folder_ + "/" + Name() + "_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    //std::ofstream out(log_name);
  //}
  {
    //auto guard_ = std::make_unique<ProfileRecord>(profiler_flag_, log_name);
    std::pair<qsl::Stack, std::vector<mlperf::QuerySample>> front;
    while (true) {
      while(!mQueuePreprocessed_.wait_dequeue_timed(front, 5) && !mStop_);
      if (mStop_) break;
      std::cout << "Model inference_at index " << index << " size " << front.first.size() << std::endl;
      auto res = model_.inference_at(which, front.first);
      QuerySamplesComplete(front.second, res[0], res[1]);
      //std::cout << "end QuerySamplesComplete size " << mQueuePreprocessed_.size_approx() << " index " << index << std::endl;
      //if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        //guard_->~ProfileRecord();
    }
    //std::cout << "exit index " << index << std::endl;
  }
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
      qsl::Stack fea_stack;
      if (preprocessor_flag_) {
        auto wav_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
        auto pre_results = preprocessor_.inference_at(which, wav_stack);
        fea_stack = qsl_.GetIValueListFrom(pre_results);
        // Test preprocessor only(response {N, C, T})
        //QuerySamplesComplete(samples, fea_stack[0].toTensor());
        //continue;
        // {N, C, T} -> {T, N, C}
        fea_stack[0] = fea_stack[0].toTensor().permute({2, 0, 1}).contiguous();
      } else {
        fea_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
      }
      auto res = model_.inference_at(which, fea_stack);
      QuerySamplesComplete(samples, res[0], res[1]);

      nIteration += 1;
      //printf("nIteration %d\n", nIteration);
      if (profiler_flag_ && profiler_iter_ != -1 && nIteration >= profiler_iter_) {
        guard_->~ProfileRecord();
        break;
      }
    }
  }
}

void RNNTSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  std::cout << "IssueQuery samples size " << samples.size() << std::endl;
  if (batch_sort_) {
    // Parallel sort samples into a queue
    mQueue_ = qsl_.Sort(samples);
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
    //std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(results[i]) << std::endl;
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

void RNNTSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const at::Tensor& results,
    const at::Tensor& result_lens) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());
  
  for (size_t i = 0; i < samples.size(); ++i) {
    auto result_lens_int = result_lens[i].item().toInt();
    //std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(results[i], result_lens_int) << std::endl;
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(results[i].data_ptr());
    responses[i].size = result_lens_int * 4;
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
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

