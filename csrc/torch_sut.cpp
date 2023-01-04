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

namespace rnnt{
ProfileRecord::ProfileRecord(
    bool is_record,
    const std::string& profiler_file) : is_record_(is_record), profiler_file_(profiler_file) {
  if (is_record_)
  {
    torch_profiler = std::make_unique<torch::autograd::profiler::RecordProfile>(profiler_file_);
  }
}

OfflineSUT::OfflineSUT(
    const std::string& sample_file,
    const std::string& model_file,
    const std::string& processor_file,
    int inter_parallel,
    int intra_parallel,
    int batch_size,
    int split_len,
    const std::string test_scenario,
    bool processor,
    const std::string& profiler_folder,
    int profiler_iter,
    int warmup_iter
  ) : qsl_(sample_file), model_(model_file), processor_(processor_file),
  nInstances_(inter_parallel), nThreadsPerInstance_(intra_parallel),
  mThreshold_(batch_size), split_len_(split_len),
  test_scenario_(test_scenario), processor_flag_(processor),
  profiler_folder_(profiler_folder), profiler_iter_(profiler_iter),
  warmup_iter_(warmup_iter) {

  batch_sort_ = (test_scenario == "Offline");

  nMaxThread_ = std::thread::hardware_concurrency();
  mHt_ = (nMaxThread_ == kmp::KMPLauncher::getMaxProc());

  // Verify nInstance_
  if ((nThreadsPerInstance_ * nInstances_) > (nMaxThread_ / (mHt_+1)))
    nInstances_ = nMaxThread_ / (mHt_ + 1) / nThreadsPerInstance_;

  std::cout << "Use HT: " << mHt_ << std::endl;
  std::cout << "Use Processor: " << processor_flag_ << std::endl;
  std::cout << "Sort samples: " << batch_sort_ << std::endl;
  std::cout << "Warmup Iteration: " << warmup_iter_ << std::endl;

  // Construct instances
  if (test_scenario_ == "Offline")
    for (int i = 0; i < nInstances_; ++ i)
      mInstances_.emplace_back(&OfflineSUT::thInstance, this, i);
}

int OfflineSUT::rootProc(int index) {
  // XXX : Assumed 2-sockets
  int part[] = {nMaxThread_, nMaxThread_ * (2 + (int)mHt_) / 4};
  auto select = index & 1;
  int root = part[select] - nThreadsPerInstance_ * ((index >> 1) + 1);

  // Dump binding info
  auto core_end = root + nThreadsPerInstance_ - 1;
  std::cout << "Binding worker " << index << " to " << root << "-" << core_end << std::endl << std::flush;

  // Assert root > 0
  return root;
}

void OfflineSUT::thInstance(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nThreadsPerInstance_);
  auto root = rootProc(index);
  auto which = index & 1;

  for (int i = 0; i < nThreadsPerInstance_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  Queue_t snippet;

  // Wait for work
  std::string log_name;
  if (profiler_iter_ > 0) {
    log_name = profiler_folder_ + "/" + Name() + "_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    std::ofstream out(log_name);
  }
  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_);
  {
    auto guard_ = std::make_unique<ProfileRecord>((profiler_iter_ > 0), log_name);
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
      auto input_stack = qsl_.AssembleSamples(std::move(indices), processor_flag_);
      if (processor_flag_) {
        std::tie(fea, fea_lens) = inferProcessor(which, input_stack);
        fea = fea.permute({2, 0, 1}).contiguous();
      } else {
        fea = input_stack[0].toTensor();
        fea_lens = input_stack[1].toTensor();
      }

      state.update(fea, fea_lens, split_len_);
      inferEncoder(which, state);
      inferDecoder(which, state);

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

void OfflineSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
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

void OfflineSUT::warmup(int which, int warmup_iter) {
  // auto start = std::chrono::high_resolution_clock::now();
  long batch_size = (long)mThreshold_;
  at::Tensor fea, fea_lens;
  rnnt::State state;
  for (int i = 0; i < warmup_iter; ++i) {
    auto wav = torch::randn({batch_size, rnnt::MAX_WAV_LEN});
    auto wav_lens = torch::full({batch_size}, rnnt::MAX_WAV_LEN, torch::kInt64);
    std::tie(fea, fea_lens) = inferProcessor(which, {wav, wav_lens});
    fea = fea.permute({2, 0, 1}).contiguous();
    state.update(fea, fea_lens, split_len_);
    inferEncoder(which, state);
    inferDecoder(which, state);
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "Warmup done. cost:" << elapsed.count() << std::endl;
}

std::tuple<at::Tensor, at::Tensor> OfflineSUT::inferProcessor(int which, qsl::Stack wav_stack) {
  auto pre_results = processor_.inference_at(which, wav_stack).toTuple()->elements();
  auto fea = pre_results[0].toTensor();
  auto fea_lens = pre_results[1].toTensor();
  // Test processor only(response {N, C, T})
  //QuerySamplesComplete(samples, fea);
  return {fea, fea_lens};
}

template <class T>
void OfflineSUT::inferEncoder(int which, T& state) {
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
}

template <class T>
void OfflineSUT::inferDecoder(int which, T& state) {
  model_.greedy_decode(which, state);
}

void OfflineSUT::QuerySamplesComplete(
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

void OfflineSUT::QuerySamplesComplete(
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

OfflineSUT::~OfflineSUT() {
  {
    std::unique_lock<std::mutex> l(mtx_);
    mStop_ = true;
    ctrl_.notify_all();
  }

  for (auto& Instance : mInstances_)
    Instance.join();
}

ServerSUT::ServerSUT(
    const std::string& sample_file,
    const std::string& model_file,
    const std::string& processor_file,
    int pre_parallel,
    int inter_parallel,
    int intra_parallel,
    int pre_batch_size,
    int batch_size,
    int split_len,
    std::string test_scenario,
    bool processor,
    const std::string& profiler_folder,
    int profiler_iter,
    int warmup_iter
  ) : OfflineSUT(sample_file, model_file, processor_file,
      inter_parallel, intra_parallel, batch_size, split_len,
      test_scenario, processor, profiler_folder, profiler_iter, warmup_iter),
      nProcessors_(pre_parallel), mProThreshold_(pre_batch_size), mQueueProcessed_(3000) {

  // Verify nInstance_
  if ((nThreadsPerInstance_ * nInstances_ + pipeline_flag_ * nProcessors_) > (nMaxThread_ / (mHt_+1)))
    nInstances_ = (nMaxThread_ / (mHt_ + 1) - pipeline_flag_ * nProcessors_) / nThreadsPerInstance_;

  // Construct instances
  mInstances_.emplace_back(&ServerSUT::thInstanceProducer, this, 0);
  for (int i = 0; i < nInstances_; ++ i)
    mInstances_.emplace_back(&ServerSUT::thInstanceConsumer, this, i);

}

int ServerSUT::rootProc(int index, int worker_type) {
  // XXX : Assumed 2-sockets
  int root = 0, core_end = 0, select = 0;
  int part[2];
  std::string type_name;
  switch (worker_type) {
    case Producer:
      root = nMaxThread_ - nProcessors_;
      core_end = root + nProcessors_ - 1;
      type_name = "Producer";
      break;
    case Consumer:
      part[0] = nMaxThread_ - pipeline_flag_ * nProcessors_;
      part[1] = nMaxThread_ * (2 + (int)mHt_) / 4 - pipeline_flag_ * (nProcessors_ / 2);
      select = index & 1;
      root = part[select] - nThreadsPerInstance_ * ((index >> 1) + 1);
      core_end = root + nThreadsPerInstance_ - 1;
      type_name = "Consumer";
      break;
    default:
      std::cout << "Unknown, worker type must be [Producer/Consumer]" << std::endl;
  }

  // Dump binding info
  std::cout << "Binding " << type_name << " worker " << index << " to " << root << "-" << core_end << std::endl << std::flush;

  // Assert root > 0
  return root;
}

void ServerSUT::thInstanceProducer(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nProcessors_);
  auto root = rootProc(index, Producer);
  auto which = index & 1;

  for (int i = 0; i < nProcessors_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  Queue_t snippet;

  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_, Producer);

  {
    while (true) {
      // critical section
      {
        std::unique_lock<std::mutex> l(mtx_);
        ctrl_.wait(l, [this] {return mStop_ || !mQueue_.empty();});

        if (mStop_) break;

        auto nPeek = std::min(mQueue_.size(), mProThreshold_);
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
      auto input_stack = qsl_.AssembleSamples(std::move(indices), processor_flag_);
      std::tie(fea, fea_lens) = inferProcessor(which, input_stack);
      auto fea_list = torch::split(fea, 1);
      auto fea_lens_list = torch::split(fea_lens, 1);
      // TODO: bulk_enqueue
      for (int i = 0; i < samples.size(); ++i) {
        mQueueProcessed_.enqueue({samples[i], fea_list[i], fea_lens_list[i]});
      }
      // std::cout << "Producer worker " << index << "::process size " << snippet.size()
      //     << ", current queue size: " << mQueueProcessed_.size_approx() << std::endl << std::flush;
      if (mQueue_.empty()) finish_processor_ = true;
    }
  }
}

void ServerSUT::thInstanceConsumer(int index) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nThreadsPerInstance_);
  auto root = rootProc(index, Consumer);
  auto which = index & 1;

  for (int i = 0; i < nThreadsPerInstance_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();
  Queue_t snippet;

  // Wait for work
  std::string log_name;
  if (profiler_iter_ > 1) {
    log_name = profiler_folder_ + "/" + Name() + "_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    std::ofstream out(log_name);
  }

  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_, Consumer);

  {
    auto guard_ = std::make_unique<ProfileRecord>((profiler_iter_ > 0), log_name);

    int nIteration = 0;
    int32_t dequeue_size;

    std::vector<mlperf::QuerySample> samples(mThreshold_);
    rnnt::PipelineState state (mThreshold_);

    while (true) {
      auto start = std::chrono::high_resolution_clock::now();
      std::vector<std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>> dequeue_list(mThreshold_);
      dequeue_size = 0;

      bool finish_dequeue = false;
      while (!mStop_ && dequeue_size == 0 && !finish_dequeue) {
        if (finish_processor_ && mQueueProcessed_.size_approx() == 0) {
          finish_dequeue = true;
          break;
        }
        dequeue_size = mQueueProcessed_.wait_dequeue_bulk_timed(dequeue_list.begin(), state.finish_size_, 0); 
      }

      if (mStop_ || (finish_dequeue && state.remain_size_ == 0)) break;

      // std::cout << "Consumer worker " << index << "::start server iteration " << nIteration
      //     << ", dequeue(" << dequeue_size << ")/require(" << state.finish_size_ << ")"
      //     << ", current queue size: " << mQueueProcessed_.size_approx() << std::endl << std::flush;

      state.update(dequeue_list, samples, dequeue_size, split_len_);
      auto last_remain_size = state.remain_size_;
      // std::cout << "finish update" << std::endl << std::flush;
      inferEncoder(which, state);
      // std::cout << "finish encoder" << std::endl << std::flush;
      inferDecoder(which, state);
      // std::cout << "finish decoder" << std::endl << std::flush;

      QuerySamplesComplete(samples, state, state.finish_idx_);
      // std::cout << "finish response" << std::endl << std::flush;

      if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        guard_->~ProfileRecord();

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      std::cout << "Consumer worker " << index << "::finish iteration " << nIteration << ", "
          << "bs(" << state.batch_size_ << ")=finish(" << state.finish_size_ << ")+remain(" << state.remain_size_
          << ")=last_remain(" << last_remain_size << ")+dequeue(" << state.dequeue_size_ << ")+padded(" << state.padded_size_ << ")"
          << ", infer model cost: " << elapsed.count() << "ms"
          << ", current queue size: " << mQueueProcessed_.size_approx() << std::endl << std::flush;

      // dequeue_list.clear();
      nIteration += 1;
    }
  }
}

void ServerSUT::warmup(int which, int warmup_iter, int worker_type) {
  // auto start = std::chrono::high_resolution_clock::now();
  long batch_size = (long)mThreshold_;
  at::Tensor wav, wav_lens;
  at::Tensor fea, fea_lens;
  auto state = (test_scenario_ == "Server") ? rnnt::State() : rnnt::PipelineState(batch_size);
  for (int i = 0; i < warmup_iter; ++i) {
    switch (worker_type) {
      case Producer:
        wav = torch::randn({batch_size, rnnt::MAX_WAV_LEN});
        wav_lens = torch::full({batch_size}, rnnt::MAX_WAV_LEN, torch::kInt64);
        std::tie(fea, fea_lens) = inferProcessor(which, {wav, wav_lens});
        break;
      case Consumer:
        fea = torch::randn({batch_size, rnnt::TRANS_INPUT_SIZE, rnnt::MAX_FEA_LEN});
        fea = fea.permute({2, 0, 1}).contiguous();
        fea_lens = torch::full({batch_size}, rnnt::MAX_FEA_LEN, torch::kInt32);
        state.update(fea, fea_lens, split_len_);
        inferEncoder(which, state);
        inferDecoder(which, state);
        break;
      default:
        std::cout << "Unknown, worker type must be [Producer/Consumer]" << std::endl;
    }
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "Warmup done. cost:" << elapsed.count() << std::endl;
}

void ServerSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const rnnt::PipelineState& state,
    const at::Tensor& finish_idx) {
  std::vector<mlperf::QuerySampleResponse> responses(state.finish_size_ - state.padded_size_);
  auto res_lens = state.res_idx_ + 1;

  size_t j = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    auto res_len = res_lens[i].item().toInt();
    if (finish_idx[i].item().toBool() && state.F_lens_[i].item().toInt() > 0) {
      // std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(state.res_[i], res_len) << std::endl << std::flush;
      responses[j].id = samples[i].id;
      responses[j].data = reinterpret_cast<uintptr_t>(state.res_[i].data_ptr());
      responses[j].size = res_len * 4;
      ++j;
    }
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

}  // namespace rnnt
