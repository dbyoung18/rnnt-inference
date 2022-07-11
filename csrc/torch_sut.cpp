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

#include "torch_sut.hpp"

RNNTOfflineSUT::RNNTOfflineSUT(
    const std::string& model_file,
    const std::string& sample_file,
    int inter_parallel,
    int intra_parallel,
    int batch, bool ht
  ) : qsl_(sample_file), model_(model_file), mThreshold_(batch),
  nProcsPerInstance_(intra_parallel), nInstances_(inter_parallel), mHt_(ht) {

  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  if (nProcsPerInstance_ * nInstances_ > nMaxProc)
    nInstances_ = nMaxProc / nProcsPerInstance_;

  // Construct instances
  for (int i = 0; i < nInstances_; ++ i)
    mInstances_.emplace_back(&RNNTOfflineSUT::thInstance, this, i);
}

//
// TODO: Use hierachy information to allocate place
//
int RNNTOfflineSUT::rootProc(int index) {
  auto nMaxProc = kmp::KMPLauncher::getMaxProc();

  // XXX : Assumed 2-sockets, HT on !!!
  int part[] = {nMaxProc, nMaxProc*(2 + (int)mHt_)/4};

  auto select = index & 1;
  auto root = part[select] - nProcsPerInstance_ * ((index>>1) + 1);

  // Assert root > 0
  return root;
}

void RNNTOfflineSUT::thInstance(int index) {
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
    auto stack = qsl_.AssembleSamples(std::move(indices));
    auto results = model_.inference_at(which, stack);
    QuerySamplesComplete(samples, results);
  }
}

void RNNTOfflineSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  std::unique_lock<std::mutex> l(mtx_);
  // Parallel sort samples into a queue
  mQueue_ = qsl_.Sort(samples);
  ctrl_.notify_one();
}

void RNNTOfflineSUT::QuerySamplesComplete(
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

std::string RNNTOfflineSUT::SequenceToString(const std::vector<int64_t>& seq) {
  std::string str = "";
  std::vector<char> labels = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
	                      'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
			      't', 'u', 'v', 'w', 'x', 'y', 'z', '\''};
  for (auto ch : seq)
    str.push_back(labels[ch]);
  return str;	
}

RNNTOfflineSUT::~RNNTOfflineSUT() {
  {
    std::unique_lock<std::mutex> l(mtx_);
    mStop_ = true;
    ctrl_.notify_all();
  }

  for (auto& Instance : mInstances_)
    Instance.join();
}

