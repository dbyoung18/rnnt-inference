#include <ATen/core/grad_mode.h>
#include <c10/util/TypeCast.h>
#include <vector>
#include <list>
#include <chrono>
#include <algorithm>
#include <condition_variable>
#include <type_traits>
#include <issue_query_controller.h>
#include <loadgen.h>
#include <logging.h>
#include <query_sample.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <cmath>

#include "torch_sut.hpp"
#include "utils.hpp"


namespace rnnt{
BaseSUT::BaseSUT(
    const std::string& sample_file,
    const std::string& model_file,
    const std::string& processor_file,
    int batch_size,
    int split_len,
    const std::string test_scenario,
    bool processor,
    const std::string& profiler_folder,
    int profiler_iter,
    int warmup_iter
  ) : qsl_(sample_file), model_(model_file), processor_(processor_file),
  mThreshold_(batch_size), split_len_(split_len),
  test_scenario_(test_scenario), processor_flag_(processor),
  profiler_folder_(profiler_folder), profiler_iter_(profiler_iter),
  warmup_iter_(warmup_iter) {

  // Get HW info
  nMaxThread_ = std::thread::hardware_concurrency();  // (mHT_ + 1) * nSockets_ * nCoresPerSocket_
  nMaxProc_ = kmp::KMPLauncher::getMaxProc();
  nSockets_ = 2;  // assume 2 sockets
  nCoresPerSocket_ = nMaxProc_ / nSockets_ / 2;
  mHt_ = (nMaxThread_ == nMaxProc_);

  batch_sort_ = (test_scenario == "Offline");
  std::cout << "Use HT: " << mHt_ << std::endl;
  std::cout << "Use Processor: " << processor_flag_ << std::endl;
  std::cout << "Sort samples: " << batch_sort_ << std::endl;
  std::cout << "Warmup Iteration: " << warmup_iter_ << std::endl;
}

void BaseSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
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

void BaseSUT::QuerySamplesComplete(
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

BaseSUT::~BaseSUT() {
  {
    std::unique_lock<std::mutex> l(mtx_);
    mStop_ = true;
    ctrl_.notify_all();
  }

  for (auto& Instance : mInstances_)
    Instance.join();
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
  ) : BaseSUT(sample_file, model_file, processor_file,
      batch_size, split_len,
      test_scenario, processor,
      profiler_folder, profiler_iter, warmup_iter),
      nInstances_(inter_parallel), nThreadsPerInstance_(intra_parallel) {

  // Verify nInstances_
  if ((nThreadsPerInstance_ * nInstances_) > (nMaxThread_ / (mHt_+1)))
    nInstances_ = nMaxThread_ / (mHt_ + 1) / nThreadsPerInstance_;

  // Construct instances & bind core(assume 2 sockets)
  int part[] = {nMaxThread_, nMaxThread_ - nCoresPerSocket_};
  std::vector<int> instance_root(nInstances_);
  std::generate(instance_root.begin(), instance_root.end(), [=, index = 0, root = 0]() mutable
      {root=part[index & 1] - nThreadsPerInstance_ * ((index >> 1) + 1); ++index; return root;});

  for (int i = 0; i < nInstances_; ++ i) {
    mInstances_.emplace_back(&OfflineSUT::thInstance, this, i, instance_root[i]);
    std::cout << "Binding instance " << i << " to " << instance_root[i] << "-" << instance_root[i] + nThreadsPerInstance_ - 1 << std::endl << std::flush;
  }
}

void OfflineSUT::warmup(int which, int warmup_iter) {
  // auto start = std::chrono::high_resolution_clock::now();
  long batch_size = (long)mThreshold_;
  at::Tensor x, x_lens;
  rnnt::State state;
  for (int i = 0; i < warmup_iter; ++i) {
    std::tie(x, x_lens) = qsl_.GenerateDummySamples(batch_size, processor_flag_);
    if (processor_flag_) {
      std::tie(x, x_lens) = processor_.forward(which, qsl::Stack {x, x_lens});
      x = x.permute({2, 0, 1}).contiguous();
    }
    state.update(x, x_lens, split_len_);
    model_.forward(which, state);
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "Warmup done. cost:" << elapsed.count() << std::endl;
}

void OfflineSUT::thInstance(int index, int root) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nThreadsPerInstance_);
  auto which = index & 1;

  for (int i = 0; i < nThreadsPerInstance_; ++ i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_);

  Queue_t snippet;

  // Wait for work
  std::string log_name;
  if (profiler_iter_ > 0) {
    log_name = profiler_folder_ + "/" + Name() + "_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    std::ofstream out(log_name);
  }
  {
    auto guard_ = std::make_unique<ProfileRecord>((profiler_iter_ > 0), log_name);
    rnnt::State state;
    size_t nIteration = 0;
    // long process_dur = 0, encode_dur = 0, decode_dur = 0;
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

      // mlperf::PerfClock::time_point iter_start = mlperf::PerfClock::now();

      std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());
      std::vector<mlperf::QuerySampleIndex> indices(samples.size());
      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
          [](mlperf::QuerySample sample) {return sample.index;});

      auto actual_batch_size = int(samples.size());
      at::Tensor fea, fea_lens;
      qsl::Stack input_stack;
      if (processor_flag_) {
        // pad T to max_len in batch
        input_stack = qsl_.AssembleSamples(std::move(indices), processor_flag_);
        std::tie(fea, fea_lens) = processor_.forward(which, input_stack);
        // auto processor_end = mlperf::PerfClock::now();
        // process_dur = get_duration(iter_start, processor_end);
        // std::cout << "Instance," << index << ",iter," << nIteration
        //     << ",max_len," << fea_lens[0].item().toInt() << ",bs," << samples.size()
        //     << ",proc," << process_dur << std::endl << std::flush;
        // BaseSUT::QuerySamplesComplete(samples, fea); continue;  // Test processor only(response {N, C, T})
        fea = fea.permute({2, 0, 1}).contiguous();
      } else {
        // pad T to max_len in batch & pad N to ensure last batch accuracy
        auto padded_batch_size = (actual_batch_size + 31) / 32 * 32;
        input_stack = qsl_.AssembleSamples(std::move(indices), processor_flag_, padded_batch_size);
        fea = input_stack[0].toTensor();
        fea_lens = input_stack[1].toTensor();
      }

      state.update(fea, fea_lens, split_len_, actual_batch_size);
      // std::cout << "finish update" << std::endl << std::flush;
      model_.encode(which, state);
      // auto encoder_end = mlperf::PerfClock::now();
      // encode_dur = get_duration(processor_end, encoder_end);
      // std::cout << "finish encode" << std::endl << std::flush;
      model_.decode(which, state);
      // auto decoder_end = mlperf::PerfClock::now();
      // decode_dur = get_duration(encoder_end, decoder_end);
      // std::cout << "finish decode" << std::endl << std::flush;

      QuerySamplesComplete(samples, state);
      // std::cout << "finish response" << std::endl << std::flush;

      nIteration += 1;
      // std::cout << "Instance," << index << ",iter," << nIteration
      //     << ",max_len," << state.f_lens_[0].item().toInt() << ",bs," << samples.size()
      //     << ",proc," << process_dur << ",enc," << encode_dur << ",dec," << decode_dur
      //     << ",lat," << get_duration(iter_start) << std::endl << std::flush;
      if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        guard_->~ProfileRecord();
    }
  }
}

void OfflineSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const rnnt::State& state) {
  std::vector<mlperf::QuerySampleResponse> responses(state.actual_batch_size_);
  auto res_lens = state.res_idx_ + 1;

  for (size_t i = 0; i < samples.size(); ++i) {
    auto res_len = res_lens[i].item().toInt();
    responses[i].id = samples[i].id;
    responses[i].data = reinterpret_cast<uintptr_t>(state.res_[i].data_ptr());
    responses[i].size = res_len * 4;
    // std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(state.res_[i], res_len) << std::endl << std::flush;
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

ServerSUT::ServerSUT(
    const std::string& sample_file,
    const std::string& model_file,
    const std::string& processor_file,
    int pro_inter_parallel,
    int pro_intra_parallel,
    int inter_parallel,
    int intra_parallel,
    int pro_batch_size,
    int batch_size,
    int split_len,
    int response_size,
    std::string test_scenario,
    bool processor,
    const std::string& profiler_folder,
    int profiler_iter,
    int warmup_iter
  ) : BaseSUT(sample_file, model_file, processor_file,
      batch_size, split_len,
      test_scenario, processor,
      profiler_folder, profiler_iter, warmup_iter),
      nProducers_(pro_inter_parallel), nThreadsPerProducer_(pro_intra_parallel),
      nConsumers_(inter_parallel), nThreadsPerConsumer_(intra_parallel),
      mProThreshold_(pro_batch_size), mProcessedQueue_(3000) {

  mResponseThreshold_ = (response_size == -1) ? mThreshold_ : response_size;

  // Verify nConsumers_
  if ((nProducers_ * nThreadsPerProducer_ + nConsumers_ * nThreadsPerConsumer_) > (nMaxThread_ / (mHt_+1)))
    nConsumers_ = (nMaxThread_ / (mHt_ + 1) - nProducers_ * nThreadsPerProducer_) / nThreadsPerConsumer_;

  // Construct instances & bind core(assume 2 sockets)
  std::vector<int> producer_root(nProducers_);
  std::vector<int> consumer_root(nConsumers_);
  // if even, staggered binding
  if (ceil(nProducers_ / nSockets_) * nThreadsPerProducer_
      + ceil(nConsumers_ / nSockets_) * nThreadsPerConsumer_ == nCoresPerSocket_) {
    int producer_part[] = {nMaxThread_ - nCoresPerSocket_ * nSockets_, nMaxThread_ - nCoresPerSocket_};
    std::generate(producer_root.begin(), producer_root.end(), [=, index = 0, root = 0]() mutable
        {root=producer_part[index & 1] + nThreadsPerProducer_ * (index >> 1); ++index; return root;});

    int consumer_part[] = {nMaxThread_,
      nMaxThread_ - nProducers_ / nSockets_ * nThreadsPerProducer_ - (nConsumers_ + nSockets_ - 1) / nSockets_ * nThreadsPerConsumer_};    
    std::generate(consumer_root.begin(), consumer_root.end(), [=, index = 0, root = 0]() mutable
        {root=consumer_part[index & 1] - nThreadsPerConsumer_ * ((index >> 1) + 1); ++index; return root;});
  } else {  // sequential binding
    int producer_start = nMaxThread_ - nCoresPerSocket_ * nSockets_;
    std::generate(producer_root.begin(), producer_root.end(), [=, index = 0, root = 0]() mutable
        {root=producer_start + nThreadsPerProducer_ * index++; return root;});

    int consumer_start = producer_start + nProducers_ * nThreadsPerProducer_;
    std::generate(consumer_root.begin(), consumer_root.end(), [=, index = 0, root = 0]() mutable
        {root=consumer_start + nThreadsPerConsumer_ * index++; return root;});
  }

  for (int i = 0; i < nProducers_; ++ i) {
    mInstances_.emplace_back(&ServerSUT::thProducer, this, i, producer_root[i]);
    std::cout << "Binding producer " << i << " to " << producer_root[i] << "-" << producer_root[i] + nThreadsPerProducer_ - 1 << std::endl << std::flush;
  }

  for (int i = 0; i < nConsumers_; ++ i) {
    mInstances_.emplace_back(&ServerSUT::thConsumer, this, i, consumer_root[i]);
    std::cout << "Binding consumer " << i << " to " << consumer_root[i] << "-" << consumer_root[i] + nThreadsPerConsumer_ - 1 << std::endl << std::flush;
  }
}

void ServerSUT::warmup(int which, int warmup_iter, int worker_type) {
  // auto start = std::chrono::high_resolution_clock::now();
  long batch_size = (long)mThreshold_;
  at::Tensor x, x_lens;
  auto state = (test_scenario_ == "Server") ? rnnt::State(mThreshold_, split_len_)
      : rnnt::PipelineState(mThreshold_, split_len_, mResponseThreshold_);
  for (int i = 0; i < warmup_iter; ++i) {
    switch (worker_type) {
      case Producer:
        if (processor_flag_) {
          std::tie(x, x_lens) = qsl_.GenerateDummySamples(batch_size, true);
          std::tie(x, x_lens) = processor_.forward(which, qsl::Stack {x, x_lens});
        }
        break;
      case Consumer:
        std::tie(x, x_lens) = qsl_.GenerateDummySamples(batch_size, false);
        x = x.permute({2, 0, 1}).contiguous();
        state.update(x, x_lens, split_len_);
        model_.forward(which, state);
        break;
      default:
        std::cout << "Unknown, worker type must be [Producer/Consumer]" << std::endl;
    }
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "Warmup done. cost:" << elapsed.count() << std::endl;
}

void ServerSUT::thProducer(int index, int root) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nThreadsPerProducer_);
  auto which = index & 1;

  for (int i = 0; i < nThreadsPerProducer_; ++i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_, Producer);

  Queue_t snippet;

  // Wait for work
  // std::string log_name;
  // if (profiler_iter_ > 1) {
  //   log_name = profiler_folder_ + "/" + Name() + "Producer_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
  //   std::ofstream out(log_name);
  // }

  {
    // auto guard_ = std::make_unique<ProfileRecord>((profiler_iter_ > 0), log_name);
    // size_t nIteration = 0;
    // long dequeue_dur = 0, process_dur = 0, enqueue_dur = 0, produce_dur = 0;
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

      // mlperf::PerfClock::time_point iter_start = mlperf::PerfClock::now();

      std::vector<mlperf::QuerySample> samples(snippet.begin(), snippet.end());
      std::vector<mlperf::QuerySampleIndex> indices(samples.size());
      std::transform(samples.cbegin(), samples.cend(), indices.begin(),
          [](mlperf::QuerySample sample) {return sample.index;});
      // auto dequeue_end = mlperf::PerfClock::now();
      // dequeue_dur = get_duration(iter_start, dequeue_end);

      at::Tensor fea, fea_lens;
      qsl::Stack input_stack;
      if (processor_flag_) {
        input_stack = qsl_.AssembleSamples(std::move(indices), processor_flag_);
        std::tie(fea, fea_lens) = processor_.forward(which, input_stack);
      } else {
        input_stack = qsl_.AssembleSamples(std::move(indices), processor_flag_, samples.size());
        fea = input_stack[0].toTensor().permute({1, 2, 0}).contiguous();  // {T, N, C} -> {N, C, T}
        fea_lens = input_stack[1].toTensor();
      }
      // auto processor_end = mlperf::PerfClock::now();
      // process_dur = get_duration(dequeue_end, processor_end);

      auto fea_list = torch::split(fea, 1);
      auto fea_lens_list = torch::split(fea_lens, 1);
      // TODO: bulk_enqueue
      for (int i = 0; i < samples.size(); ++i) {
        mProcessedQueue_.enqueue({samples[i], fea_list[i], fea_lens_list[i]});
      }
      // auto enqueue_end = mlperf::PerfClock::now();
      // enqueue_dur = get_duration(process_end, enqueue_end);
      // produce_dur = get_duration(iter_start, enqueue_end);

      // if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
      //   guard_->~ProfileRecord();

      // nIteration += 1;
      // auto latency = get_latency(samples[0]);
      // std::cout << "Producer," << index << ",iter," << nIteration
      //     << ",in," << samples.size() << ",out," << fea_lens_list.size()
      //     << ",lqueue," << mQueue_.size() << ",pqueue," << mProcessedQueue_.size_approx()
          // << ",deq," << dequeue_dur << ",proc," << process_dur << ",enq," << enqueue_dur
          // << ",prod," << produce_dur << ",lat," << get_latency(samples[0])
          // << std::endl << std::flush;
      // BaseSUT::QuerySamplesComplete(samples, fea);  // Test processor only(response {N, C, T})
      // if (mQueue_.empty()) finish_enqueue_ = true;
    }
  }
}

void ServerSUT::thConsumer(int index, int root) {
  at::NoGradGuard no_grad;

  kmp::KMPLauncher thCtrl;
  std::vector<int> places (nThreadsPerConsumer_);
  auto which = index & 1;

  for (int i = 0; i < nThreadsPerConsumer_; ++i)
    places[i] = (root + i);

  thCtrl.setAffinityPlaces(places).pinThreads();

  if (warmup_iter_ > 0)
    warmup(which, warmup_iter_, Consumer);

  Queue_t snippet;

  // Wait for work
  std::string log_name;
  if (profiler_iter_ > 1) {
    log_name = profiler_folder_ + "/" + Name() + "Consumer_" + std::to_string(index) + "_" + std::to_string(which) + ".json";
    std::ofstream out(log_name);
  }

  {
    auto guard_ = std::make_unique<ProfileRecord>((profiler_iter_ > 0), log_name);

    int32_t dequeue_size;

    std::vector<mlperf::QuerySample> samples(mThreshold_);
    rnnt::PipelineState state (mThreshold_, split_len_, mResponseThreshold_);

    int nIteration = 0;
    // long dequeue_dur = 0, batch_dur = 0, encode_dur = 0, decode_dur = 0, response_dur = 0;
    while (true) {
      // mlperf::PerfClock::time_point iter_start = mlperf::PerfClock::now();
      std::vector<std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>> dequeue_list(mThreshold_);
      dequeue_size = 0;

      bool finish_dequeue = false;
      while (!mStop_ && !finish_dequeue && dequeue_size == 0 && state.finish_size_ != 0) {
        // no new samples left
        if (finish_enqueue_ && mProcessedQueue_.size_approx() == 0) {
          finish_dequeue = true;
          break;
        }
        dequeue_size = mProcessedQueue_.wait_dequeue_bulk_timed(dequeue_list.begin(), state.finish_size_, 0);
        if (state.remain_size_ != 0)
          break;
      }
      // auto dequeue_end = mlperf::PerfClock::now();
      // dequeue_dur = get_duration(iter_start, dequeue_end);

      if (mStop_ || (finish_dequeue && state.remain_size_ == 0)) {
        std::cout << "Consumer " << index << " finish iteration " << nIteration << std::endl << std::flush;
        break;
      }

      // std::cout << "Consumer," << index << ",iter," << nIteration
      //     << ",dequeue," << dequeue_size << ",require," << state.finish_size_
      //     << ",queue," << mProcessedQueue_.size_approx() << std::endl << std::flush;

      state.update(dequeue_list, samples, dequeue_size, split_len_);
      // auto batch_end = mlperf::PerfClock::now();
      // batch_dur = get_duration(dequeue_end, batch_end);
      // std::cout << "finish update" << std::endl << std::flush;
      model_.encode(which, state);
      // auto encoder_end = mlperf::PerfClock::now();
      // encode_dur = get_duration(batch_end, encoder_end);
      // std::cout << "finish encode" << std::endl << std::flush;
      model_.decode(which, state);
      // auto decoder_end = mlperf::PerfClock::now();
      // decode_dur = get_duration(encoder_end, decoder_end);
      // std::cout << "finish decode" << std::endl << std::flush;

      QuerySamplesComplete(samples, state);
      // auto response_end = mlperf::PerfClock::now();
      // response_dur = get_duration(response_end, decoder_end);
      // std::cout << "finish response" << std::endl << std::flush;

      if (profiler_iter_ != -1 && nIteration >= profiler_iter_)
        guard_->~ProfileRecord();

      // std::cout << "Consumer," << index << ",iter," << nIteration
      //     << ",infer," << state.batch_size_ << ",finish," << state.finish_size_ << ",remain," << state.remain_size_
      //     << ",pad," << state.padded_size_ << ",dequeue," << state.dequeue_size_
      //     << ",lqueue," << mQueue_.size() << ",pqueue," << mProcessedQueue_.size_approx()
      //     // << ",deq," << dequeue_dur << ",batch," << batch_dur << ",enc," << encode_dur << ",dec," << decode_dur
      //     // << ",response," << response_dur
      //     << std::endl << std::flush;

      // dequeue_list.clear();
      nIteration += 1;
    }
  }
}

void ServerSUT::QuerySamplesComplete(
    const std::vector<mlperf::QuerySample>& samples,
    const rnnt::PipelineState& state) {
  std::vector<mlperf::QuerySampleResponse> responses(state.finish_size_ - state.padded_size_);
  auto res_lens = state.res_idx_ + 1;

  size_t j = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    auto res_len = res_lens[i].item().toInt();
    if (state.finish_idx_[i].item().toBool() && state.F_lens_[i].item().toInt() > 0) {
      responses[j].id = samples[i].id;
      responses[j].data = reinterpret_cast<uintptr_t>(state.res_[i].data_ptr());
      responses[j].size = res_len * 4;
      ++j;
      auto latency = get_latency(samples[i]);
      if (latency >= 1000)
        std::cout << "finish sample " << samples[i].id << ", len " << state.F_lens_[i].item().toInt() << " ,cost " << latency << " ms\n" << std::flush;
      // std::cout << samples[i].index << "::" << models::TorchModel::sequence_to_string(state.res_[i], res_len) << std::endl << std::flush;
    }
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
}

}  // namespace rnnt
