#pragma once

#include <cassert>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <atomic>

#include <query_sample.h>
#include <condition_variable>
#include <system_under_test.h>
#include <query_sample_library.h>

#include "kmp_launcher.hpp"
#include "rnnt_qsl.hpp"
#include "rnnt_model.hpp"
#include "rnnt_preprocessor.hpp"
#include "blockingconcurrentqueue.h"
#include <torch/csrc/autograd/profiler_legacy.h>

class ProfileRecord {
public:
  ProfileRecord (bool is_record, const std::string& profiler_file);
  virtual ~ProfileRecord(){};

private:
  bool is_record_;
  std::string profiler_file_;
  std::unique_ptr<torch::autograd::profiler::RecordProfile> torch_profiler;
};


class RNNTSUT : public mlperf::SystemUnderTest {
  using Queue_t = std::list<mlperf::QuerySample>;
  using Map_t = std::pair<mlperf::ResponseId, std::tuple<mlperf::QuerySample, rnnt::TensorVector, rnnt::TensorVector>>;
public:
  // configure inter parallel and intra paralel
  RNNTSUT (
      const std::string& model_file,
      const std::string& samples_file,
      const std::string& preprocessor_file,
      int pre_parallel,
      int inter_parallel,
      int intra_parallel,
      int pre_batch_size,
      int batch_size,
      long split_len = -1,
      bool enable_bf16 = true,
      std::string test_scenario = "Offline",
      bool preprocessor = true,
      bool profiler = false,
      const std::string& profiler_foler = "",
      int profiler_iter = -1,
      int warmup_iter = -1
  );

  ~RNNTSUT ();

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

  void FlushQueries() override {}

  const std::string& Name() override {
    static const std::string name("RNN-T_" + test_scenario_ + "_SUT");
    return name;
  }

  static void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const at::Tensor& results
  );

  static void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const at::Tensor& results,
      const at::Tensor& result_lens
  );

  void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const at::Tensor& results,
      const at::Tensor& result_lens,
      const at::Tensor& finish_idx);

  mlperf::QuerySampleLibrary* GetQSL() {
    return &qsl_;
  }

private:
  qsl::RNNTQuerySampleLibrary qsl_;
  models::TorchModel model_;
  models::AudioPreprocessor preprocessor_;

  std::condition_variable ctrl_;
  std::mutex mtx_;

  Queue_t mQueue_;
  bool mStop_ {false};

  std::vector<std::thread> mInstances_;
  int nPreprocessors_;
  int nInstances_;
  int nProcsPerInstance_;
  // Control over max samples a instance will peek
  size_t mPreThreshold_;
  size_t mThreshold_;
  long split_len_;
  bool enable_bf16_;
  bool mHt_;
  int nMaxProc_;
  std::string test_scenario_;
  bool preprocessor_flag_;
  moodycamel::BlockingConcurrentQueue<std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>> mQueuePreprocessed_;
  // std::unique_ptr<ProfileRecord> guard_;
  bool profiler_flag_;
  std::string profiler_folder_;
  int profiler_iter_;
  int warmup_iter_;
  bool batch_sort_;  // Offline only
  bool pipeline_flag_;  // Server only

  int rootProc(int index, bool model_worker);
  void thInstance(int index);
  void thInstancePreprocessor(int index);
  void thInstanceModel(int index);
  void warmup(int which, int warmup_iter = 3);
  std::tuple<at::Tensor, at::Tensor> inferPreprocessor(int which, qsl::Stack wav_stack);
  void inferModel(int which, rnnt::State state);
};

