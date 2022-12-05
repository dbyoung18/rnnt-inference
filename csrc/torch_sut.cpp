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

void sleep_thread(int duration) {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));
  return;
}

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
    std::string test_scenario,
    bool preprocessor,
    bool profiler,
    const std::string& profiler_folder,
    int profiler_iter,
    int warmup_iter
  ) : qsl_(sample_file), model_(model_file), preprocessor_(preprocessor_file),
  nPreprocessors_(pre_parallel), nInstances_(inter_parallel), nProcsPerInstance_(intra_parallel),
  mPreThreshold_(pre_batch_size), mThreshold_(batch_size), split_len_(split_len),
  test_scenario_(test_scenario), preprocessor_flag_(preprocessor), mQueuePreprocessed_(3000),
  profiler_flag_(profiler), profiler_folder_(profiler_folder), profiler_iter_(profiler_iter),
  warmup_iter_(warmup_iter) {

  batch_sort_ = (test_scenario == "Offline");
  pipeline_flag_ = (test_scenario == "Server") && preprocessor_flag_;

  nMaxProc_ = std::thread::hardware_concurrency();
  mHt_ = (nMaxProc_ == kmp::KMPLauncher::getMaxProc());

  // Verify nInstance_
  if ((nProcsPerInstance_ * nInstances_ + pipeline_flag_ * nPreprocessors_) > (nMaxProc_ / (mHt_+1)))
    nInstances_ = (nMaxProc_ / (mHt_ + 1) - pipeline_flag_ * nPreprocessors_) / nProcsPerInstance_;

  std::cout << "Use HT: " << mHt_ << std::endl;
  std::cout << "Use Preprocessor: " << preprocessor_flag_ << std::endl;
  std::cout << "Use Pipeline: " << pipeline_flag_ << std::endl;
  std::cout << "Sort samples: " << batch_sort_ << std::endl;
  std::cout << "Warmup Iteration: " << warmup_iter_ << std::endl;

  // Construct instances
  if (pipeline_flag_) {
    mInstances_.emplace_back(&RNNTSUT::thInstancePreprocessor, this, 0);
    for (int i = 0; i < nInstances_; ++ i)
      mInstances_.emplace_back(&RNNTSUT::thInstanceModel, this, i);
  } else {
    for (int i = 0; i < nInstances_; ++ i)
      mInstances_.emplace_back(&RNNTSUT::thInstance, this, i);
  }

}

int RNNTSUT::rootProc(int index, bool model_worker) {
  // XXX : Assumed 2-sockets
  int root;
  if (model_worker) {
    int part[] = {nMaxProc_ - pipeline_flag_ * nPreprocessors_,
      nMaxProc_ * (2 + (int)mHt_) / 4 - pipeline_flag_ * (nPreprocessors_ / 2)};
    auto select = index & 1;
    root = part[select] - nProcsPerInstance_ * ((index >> 1) + 1);
  } else {
    root = nMaxProc_ - nPreprocessors_;
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

  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_, Processor);

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

      at::Tensor fea, fea_lens;
      auto input_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
      std::tie(fea, fea_lens) = inferPreprocessor(which, input_stack);
      auto fea_list = torch::split(fea, 1);
      auto fea_lens_list = torch::split(fea_lens, 1);
      // TODO: bulk_enqueue
      for (int i = 0; i < samples.size(); ++i) {
        mQueuePreprocessed_.enqueue({samples[i], fea_list[i], fea_lens_list[i]});
      }
      // std::cout << "preprocessor worker " << index << "::preprocess size " << snippet.size()
      //     << ", current queue size: " << mQueuePreprocessed_.size_approx() << std::endl << std::flush;
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
  std::string log_name;
  if (profiler_flag_) {
    log_name = profiler_folder_ + "/" + Name() + "_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    std::ofstream out(log_name);
  }

  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_, Model);

  {
    auto guard_ = std::make_unique<ProfileRecord>(profiler_flag_, log_name);

    int nIteration = 0;
    int32_t dequeue_size;

    std::vector<mlperf::QuerySample> samples;
    samples.reserve(mThreshold_);
    rnnt::PipelineState state (mThreshold_);

    while (true) {
      auto start = std::chrono::high_resolution_clock::now();
      std::vector<std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>> dequeue_list(mThreshold_);
      dequeue_size = 0;

      while (dequeue_size == 0 && !mStop_)
        dequeue_size = mQueuePreprocessed_.wait_dequeue_bulk_timed(dequeue_list.begin(), state.finish_size_, 0);

      std::cout << "model worker " << index << "::start server iteration " << nIteration
          << ", dequeue_size/require_size " << dequeue_size << "/" << state.finish_size_
          << ", current queue size: " << mQueuePreprocessed_.size_approx() << std::endl << std::flush;
      if (mStop_) break;

      state.update(dequeue_list, samples, dequeue_size, split_len_);
      auto pre_remain_size = state.remain_size_;
      // std::cout << "finish update" << std::endl << std::flush;
      inferModel(which, state);

      QuerySamplesComplete(samples, state, state.finish_idx_);
      // std::cout << "finish response" << std::endl << std::flush;

      if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        guard_->~ProfileRecord();

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      std::cout << "model worker " << index << "::finish iteration " << nIteration << ", "
          // << "infer=finish+remain=pre_remain+dequeue+padded:"
          << state.batch_size_ << "=" << state.finish_size_ << "+" << state.remain_size_ << "="
          << pre_remain_size << "+" << state.dequeue_size_ << "+" << state.padded_size_
          << ", infer model cost: " << elapsed.count() << "ms"
          << ", current queue size: " << mQueuePreprocessed_.size_approx() << std::endl << std::flush;

      // dequeue_list.clear();
      samples.clear();
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
  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_, EndToEnd);
  {
    auto guard_ = std::make_unique<ProfileRecord>(profiler_flag_, log_name);
    size_t nIteration = 0;
    rnnt::State state;
    while (true) {
      // auto start = std::chrono::high_resolution_clock::now();
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
      std::vector<mlperf::QuerySampleIndex> indices(samples.size());
      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
          [](mlperf::QuerySample sample) {return sample.index;});

      at::Tensor fea, fea_lens;
      auto input_stack = qsl_.AssembleSamples(std::move(indices), preprocessor_flag_);
      if (preprocessor_flag_) {
        std::tie(fea, fea_lens) = inferPreprocessor(which, input_stack);
        fea = fea.permute({2, 0, 1}).contiguous();
      } else {
        fea = input_stack[0].toTensor();
        fea_lens = input_stack[1].toTensor();
      }
      // do inference
      state.update(fea, fea_lens, split_len_);
      inferModel(which, state);

      QuerySamplesComplete(samples, state);

      // auto end = std::chrono::high_resolution_clock::now();
      // std::chrono::duration<double, std::milli> elapsed = end - start;
      // std::cout << "Warmup done. cost:" << elapsed.count() << std::endl;

      nIteration += 1;
      if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        guard_->~ProfileRecord();
    }
  }
}

void RNNTSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  if (batch_sort_) {
    // Parallel sort samples into a queue
    mQueue_ = qsl_.Sort(samples);
  } else {
    for (auto sample : samples)
      mQueue_.emplace_back(sample);
  }
  ctrl_.notify_one();
}

void RNNTSUT::warmup(int which, int warmup_iter, int worker_type) {
  // auto start = std::chrono::high_resolution_clock::now();
  long batch_size = (long)mThreshold_;
  at::Tensor wav, wav_lens;
  at::Tensor fea, fea_lens;
  auto state = (test_scenario_ == "Server") ? rnnt::State() : rnnt::PipelineState(batch_size);
  for (int i = 0; i < warmup_iter; ++i) {
    switch (worker_type) {
      case Processor:
        wav = torch::randn({batch_size, rnnt::MAX_WAV_LEN});
        wav_lens = torch::full({batch_size}, rnnt::MAX_WAV_LEN, torch::kInt64);
        std::tie(fea, fea_lens) = inferPreprocessor(which, {wav, wav_lens});
        break;
      case Model:
        fea = torch::randn({batch_size, rnnt::TRANS_INPUT_SIZE, rnnt::MAX_FEA_LEN});
        fea = fea.permute({2, 0, 1}).contiguous();
        fea_lens = torch::full({batch_size}, rnnt::MAX_FEA_LEN, torch::kInt32);
        state.update(fea, fea_lens, split_len_);
        inferModel(which, state);
        break;
      case EndToEnd:
        wav = torch::randn({batch_size, rnnt::MAX_WAV_LEN});
        wav_lens = torch::full({batch_size}, rnnt::MAX_WAV_LEN, torch::kInt64);
        std::tie(fea, fea_lens) = inferPreprocessor(which, {wav, wav_lens});
        fea = fea.permute({2, 0, 1}).contiguous();
        state.update(fea, fea_lens, split_len_);
        inferModel(which, state);
        break;
      default:
        std::cout << "Unknown, worker type must be [Processor/Model/EndToEnd]" << std::endl;
    }
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "Warmup done. cost:" << elapsed.count() << std::endl;
}

std::tuple<at::Tensor, at::Tensor> RNNTSUT::inferPreprocessor(int which, qsl::Stack wav_stack) {
  auto pre_results = preprocessor_.inference_at(which, wav_stack).toTuple()->elements();
  // {N, C, T} -> {T, N, C}
  auto fea = pre_results[0].toTensor();
  auto fea_lens = pre_results[1].toTensor();
  // Test preprocessor only(response {N, C, T})
  //QuerySamplesComplete(samples, fea);
  return {fea, fea_lens};
}

template <class T>
void RNNTSUT::inferModel(int which, T& state) {
  if (split_len_ != -1) {
    // accumulate transcription
    rnnt::TensorVector fi_list;
    fi_list.reserve(rnnt::HALF_MAX_FEA_LEN);
    while(state.next()) {
      model_.transcription_encode(which, state);
      fi_list.emplace_back(state.f_);
    }
    state.f_ = torch::cat(fi_list, 0);
  } else {
    model_.transcription_encode(which, state);
  }
  state.f_lens_ = torch::ceil(state.infer_lens_ / rnnt::STACK_TIME_FACTOR).to(torch::kInt32);
  // std::cout << "finish encode" << std::endl << std::flush;

  model_.greedy_decode(which, state);
  // std::cout << "finish decode" << std::endl << std::flush;
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
    const rnnt::State& state) {
  std::vector<mlperf::QuerySampleResponse> responses(samples.size());
  auto res_lens = state.res_idx_ + 1;

  for (size_t i = 0; i < samples.size(); ++i) {
    auto res_len = res_lens[i].item().toInt();
    // std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(state.res_[i], res_len) << std::endl << std::flush;
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(state.res_[i].data_ptr());
    responses[i].size = res_len * 4;
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

// for Server
void RNNTSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const rnnt::PipelineState& state,
    const at::Tensor& finish_idx) {
  std::vector<mlperf::QuerySampleResponse> responses(state.finish_size_ - state.padded_size_);
  auto res_lens = state.res_idx_ + 1;

  for (size_t i = 0; i < samples.size(); ++i) {
    auto res_len = res_lens[i].item().toInt();
    if (finish_idx[i].item().toBool() && state.F_lens_[i].item().toInt() > 0) {
      // std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(state.res_[i], res_len) << std::endl << std::flush;
      responses[i].id = samples[i].id;
      responses[i].data = reinterpret_cast<uintptr_t>(state.res_[i].data_ptr());
      responses[i].size = res_len * 4;
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
