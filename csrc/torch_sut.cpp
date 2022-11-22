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
#include <thread>
#include <chrono>

#include "torch_sut.hpp"

using namespace torch::indexing;

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
    int pre_parallel,
    int inter_parallel,
    int intra_parallel,
    int pre_batch_size,
    int batch_size,
    long split_len,
    bool enable_bf16,
    bool ht,
    std::string test_scenario,
    bool preprocessor,
    bool profiler,
    const std::string& profiler_folder,
    int profiler_iter
  ) : qsl_(sample_file), model_(model_file, enable_bf16), preprocessor_(preprocessor_file),
  nPreprocessors_(pre_parallel), nInstances_(inter_parallel), nProcsPerInstance_(intra_parallel),
  mPreThreshold_(pre_batch_size), mThreshold_(batch_size), split_len_(split_len), enable_bf16_(enable_bf16),
  mHt_(ht), test_scenario_(test_scenario), preprocessor_flag_(preprocessor), mQueuePreprocessed_(3000),
  profiler_flag_(profiler), profiler_folder_(profiler_folder), profiler_iter_(profiler_iter) {

  batch_sort_ = (test_scenario == "Offline");
  pipeline_flag_ = (test_scenario == "Server") && preprocessor_flag_;

  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  // Verify nInstance_
  if ((nProcsPerInstance_ * nInstances_ + pipeline_flag_ * nPreprocessors_) > (nMaxProc / (mHt_+1)))
    nInstances_ = (nMaxProc / (mHt_ + 1) - pipeline_flag_ * nPreprocessors_) / nProcsPerInstance_;

  // Construct instances
  if (pipeline_flag_) {
    std::cout << "Use pipeline mode!" << std::endl;
    mInstances_.emplace_back(&RNNTSUT::thInstancePreprocessor, this, 0);
    for (int i = 0; i < nInstances_; ++ i)
      mInstances_.emplace_back(&RNNTSUT::thInstanceModel, this, i);
  } else {
    for (int i = 0; i < nInstances_; ++ i)
      mInstances_.emplace_back(&RNNTSUT::thInstance, this, i);
  }

}

//
// TODO: Use hierachy information to allocate place
//
int RNNTSUT::rootProc(int index, bool model_worker = true) {
  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  // XXX : Assumed 2-sockets, HT on !!!
  int root;
  if (model_worker) {
    int part[] = {nMaxProc - pipeline_flag_ * nPreprocessors_,
      nMaxProc *(2 + (int)mHt_)/4 - pipeline_flag_ * (nPreprocessors_/2)};
    auto select = index & 1;
    root = part[select] - nProcsPerInstance_ * ((index>>1) + 1);
  } else {
    root = nMaxProc - nPreprocessors_;
  }

  // Dump binding info
  auto worker_type = model_worker ? "model" : "preprocessor";
  auto core_end = model_worker ? (root + nProcsPerInstance_ - 1) : (root + nPreprocessors_ - 1);
  std::cout << "Binding " << worker_type << " worker " << index << " to " << root << "-" << core_end << std::endl << std::flush;

  // Assert root > 0
  return root;
}

void RNNTSUT::thInstancePreprocessor(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nPreprocessors_);
  auto root = rootProc(index, false);
  auto which = index & 1;

  for (int i = 0; i < nPreprocessors_; ++ i)
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

        auto nPeek = std::min(mQueue_.size(), mPreThreshold_);
        auto it = mQueue_.begin();
        // XXX: pointer chaser, choose better solution
        std::advance(it, nPeek);
        snippet.clear();
        snippet.splice(snippet.begin(), mQueue_, mQueue_.begin(), it);

        // if (!mQueue_.empty()) Offline never empty
        ctrl_.notify_one();
      }

      std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());
      std::vector<mlperf::QuerySampleIndex> indices(samples.size());
      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
          [](mlperf::QuerySample sample) {return sample.index;});

      auto wav_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
      auto pre_results = preprocessor_.inference_at(which, wav_stack);
      auto fea_stack = qsl_.GetIValueListFrom(pre_results);
      // Test preprocessor only(response {N, C, T})
      // QuerySamplesComplete(samples, fea_stack[0].toTensor());
      // continue;
      // {Np, C, T} -> Np * {1, C, T}
      auto fea_list = torch::split(fea_stack[0].toTensor(), 1);
      auto fea_len_list = torch::split(fea_stack[1].toTensor(), 1);
      // TODO: bulk_enqueue
      for (int i = 0; i < samples.size(); ++i) {
        mQueuePreprocessed_.enqueue({samples[i], fea_list[i], fea_len_list[i]});
      }
      std::cout << "preprocessor worker " << index << "::preprocess size " << snippet.size()
          << ", current queue size: " << mQueuePreprocessed_.size_approx() << std::endl << std::flush;
    }
  }
}

void sleep_thread(int duration) {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));
  return;
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
  std::string log_name;
  if (profiler_flag_) {
    log_name = profiler_folder_ + "/" + Name() + "_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    std::ofstream out(log_name);
  }
  {
    auto guard_ = std::make_unique<ProfileRecord>(profiler_flag_, log_name);
    sleep_thread(1000);

    size_t nIteration = 0;
    std::vector<mlperf::QuerySample> samples;
    samples.reserve(mThreshold_);

    int32_t dequeue_size = 0;
    // std::vector<std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>> dequeue_list(mThreshold_);
    rnnt::State state (mThreshold_, enable_bf16_);
    rnnt::TensorVector f, f_lens;
    f.reserve(mThreshold_);
    f_lens.reserve(mThreshold_);

    while (true) {
      auto start = std::chrono::high_resolution_clock::now();
      std::vector<std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>> dequeue_list(mThreshold_);
      std::cout << "--dequeue_list.size:" << dequeue_list.size() << ",dequeue_list.capacity:" << dequeue_list.capacity() << ",finish_size:" << state.finish_size_ << std::endl;

      // for (int i = 0; i < dequeue_size; ++i) {
      //   auto sample = std::get<0>(dequeue_list[i]);
      //   auto fi = std::get<1>(dequeue_list[i]).permute({2, 0, 1}).contiguous();
      //   auto fi_len = std::get<2>(dequeue_list[i]);
      //   std::cout << i << "::" << sample.index << ", fi.shape:" << fi.sizes() << std::endl;
      // }

      while (dequeue_size == 0 && !mStop_)
        dequeue_size = mQueuePreprocessed_.wait_dequeue_bulk_timed(dequeue_list.begin(), state.finish_size_, 0);
      std::cout << "model worker " << index << "::start server iteration " << nIteration
          << ", dequeue_size/require_size " << dequeue_size << "/" << state.finish_size_
          << ", current queue size: " << mQueuePreprocessed_.size_approx() << std::endl << std::flush;
      if (mStop_) break;

      // insert new samples
      for (int i = 0; i < dequeue_size; ++i) {
        auto sample = std::get<0>(dequeue_list[i]);
        auto fi = std::get<1>(dequeue_list[i]).permute({2, 0, 1}).contiguous();
        auto fi_len = std::get<2>(dequeue_list[i]);
        samples.emplace_back(sample);
        f.emplace_back(fi);
        f_lens.emplace_back(fi_len);
      }

      state.update(f, f_lens, split_len_);
      // inference
      // sleep_thread(500);
      if (split_len_ != -1) {
        // TODO: accumulate transcription for Server?
        while(state.next()) {
          model_.transcription_encode(which, state);
          model_.greedy_decode(which, state);
          if (state.finish_size_ != 0) {
            QuerySamplesComplete(samples, state.res_, state.res_idx_+1, state.finish_idx_);
            break;
          }
        }
      } else {
        model_.transcription_encode(which, state);
        model_.greedy_decode(which, state);
        QuerySamplesComplete(samples, state.res_, state.res_idx_+1);
      }

      samples.clear();
      state.reset();
      f.clear();
      f_lens.clear();

      if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        guard_->~ProfileRecord();

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      std::cout << "model worker " << index << "::finish server iteration " << nIteration
          << ", infer bs " << state.batch_size_
          << ", fea shape: " << state.f_.sizes()
          << ", cost: " << elapsed.count() << "ms\n" << std::flush;
      nIteration += 1;
    }
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
      at::Tensor fea, fea_lens;
      if (preprocessor_flag_) {
        auto wav_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
        auto pre_results = preprocessor_.inference_at(which, wav_stack);
        fea_stack = qsl_.GetIValueListFrom(pre_results);
        // Test preprocessor only(response {N, C, T})
        //QuerySamplesComplete(samples, fea_stack[0].toTensor());
        //continue;
        // {N, C, T} -> {T, N, C}
        fea = fea_stack[0].toTensor().permute({2, 0, 1}).contiguous();
        fea_lens = fea_stack[1].toTensor();
      } else {
        fea_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
        fea = fea_stack[0].toTensor();
        fea_lens = fea_stack[1].toTensor();
      }
      // do inference
      auto actual_batch_size = fea_lens.size(0);
      rnnt::State state (actual_batch_size, enable_bf16_);
      state.update(fea, fea_lens, split_len_);

      if (split_len_ != -1) {
        // accumulate transcription
        rnnt::TensorVector fi_list;
        fi_list.reserve(rnnt::HALF_MAX_LEN);
        while(state.next()) {
          model_.transcription_encode(which, state);
          fi_list.emplace_back(state.f_);
        }
        state.f_ = torch::cat(fi_list, 0);
        state.f_lens_ = torch::ceil(state.F_lens_ / 2).to(torch::kInt32);
      } else {
        model_.transcription_encode(which, state);
      }

      model_.greedy_decode(which, state);

      QuerySamplesComplete(samples, state.res_, state.res_idx_+1);
      nIteration += 1;
      if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        guard_->~ProfileRecord();
    }
  }
}

void RNNTSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  //std::cout << "IssueQuery samples size " << samples.size() << std::endl;
  if (batch_sort_) {
    std::cout << "use batch sort" << std::endl;
    // Parallel sort samples into a queue
    mQueue_ = qsl_.Sort(samples);
  } else {
    for (auto sample : samples)
      mQueue_.emplace_back(sample);
  }
  ctrl_.notify_one();
}

// for Preprocessor
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

// for Offline
void RNNTSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const at::Tensor& results,
    const at::Tensor& result_lens) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    auto result_lens_int = result_lens[i].item().toInt();
    // std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(results[i], result_lens_int) << std::endl << std::flush;
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(results[i].data_ptr());
    responses[i].size = result_lens_int * 4;
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

// for Server
void RNNTSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const at::Tensor& results,
    const at::Tensor& result_lens,
    const at::Tensor& finish_idx) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    auto result_lens_int = result_lens[i].item().toInt();
    if (finish_idx[i].item().toBool()) {
      // std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(results[i], result_lens_int) << std::endl << std::flush;
      responses[i].id = samples[i].id;
      responses[i].data = reinterpret_cast<uintptr_t>(results[i].data_ptr());
      responses[i].size = result_lens_int * 4;
    }
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

