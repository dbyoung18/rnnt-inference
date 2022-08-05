#pragma once

#include <cassert>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>

#include <query_sample.h>
#include <condition_variable>
#include <system_under_test.h>
#include <query_sample_library.h>

#include "kmp_launcher.hpp"
#include "rnnt_qsl.hpp"
#include "rnnt_model.hpp"
#include "rnnt_preprocessor.hpp"
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
public:
  // configure inter parallel and intra paralel
  RNNTSUT (
      const std::string& model_file,
      const std::string& samples_file,
      const std::string& preprocessor_file,
      int inter_parallel,
      int intra_parallel,
      int batch,
      bool ht = true,
      bool profiler = false,
      const std::string& profiler_foler = "",
      bool preprocessor = true,
      std::string test_scenario = "Offline",
      int perf_count = 2513
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
      const std::vector<std::vector<int64_t>>& results
  );

  static void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const at::Tensor& results
  );

  static std::string SequenceToString(const std::vector<int64_t>& seq);

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

  // Control over max samples a instance will peek
  size_t mThreshold_;

  std::vector<std::thread> mInstances_;
  int nProcsPerInstance_;
  int nInstances_;
  bool mHt_;
  bool profiler_flag_;
  std::string profiler_folder_;
  // std::unique_ptr<ProfileRecord> guard_;
  bool preprocessor_flag_;
  std::string test_scenario_;
  size_t perf_count_;

  int rootProc(int index);
  void thInstance(int index);
};

