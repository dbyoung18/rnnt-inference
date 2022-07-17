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

class RNNTOfflineSUT : public mlperf::SystemUnderTest {

  using Queue_t = std::list<mlperf::QuerySample>;
  // using Queue_t = std::forward_list<mlperf::QuerySample>
  // using Queue_t = std::deque<mlperf::QuerySample>;
public:
  // configure inter parallel and intra paralel
  RNNTOfflineSUT (
      const std::string& model_file,
      const std::string& samples_file,
      int inter_parallel,
      int intra_parallel,
      int batch, bool ht = true
  );

  ~RNNTOfflineSUT ();

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

  void FlushQueries() override {}

  const std::string& Name() const override {
    static const std::string name("RNN-T Offline");
    return name;
  }

  static void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const std::vector<std::vector<int64_t>>& results
  );

  static std::string SequenceToString(const std::vector<int64_t>& seq);

  mlperf::QuerySampleLibrary* GetQSL() {
    return &qsl_;
  }

private:
  qsl::RNNTQuerySampleLibrary qsl_;
  models::TorchModel model_;

  std::condition_variable ctrl_;
  std::mutex mtx_;

  Queue_t mQueue_;
  bool mStop_ {false};

  // Control over max samples a instance will peek
  size_t mThreshold_;
  size_t slength_ {500};

  std::vector<std::thread> mInstances_;
  int nProcsPerInstance_;
  int nInstances_;
  bool mHt_;

  int rootProc(int index);
  void thInstance(int index);
};

