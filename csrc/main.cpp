#include <loadgen.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>

#include "cxxopts.hpp"
#include "torch_sut.hpp"
#include "test_settings.h"

std::map<std::string, mlperf::TestScenario> scenario_map = {
  {"Offline", mlperf::TestScenario::Offline},
  {"Server", mlperf::TestScenario::Server}};

int main(int argc, char **argv) {
  cxxopts::Options opts (
    "rnnt_inference", "MLPerf Benchmark, RNN-T Inference");
  // opts.allow_unrecognised_options();
  opts.add_options(
      "", {{"m,model_file", "Torch Model File", cxxopts::value<std::string>()},

           {"s,sample_file", "LibriSpeech Sample File",
            cxxopts::value<std::string>()},

           {"preprocessor_file", "Audio Preprocessor File",
            cxxopts::value<std::string>()},

           {"pre_parallel", "Instance Number of preprocessor(pipeline mode only)",
            cxxopts::value<int>()->default_value("8")},

           {"n,inter_parallel", "Instance Number",
            cxxopts::value<int>()->default_value("1")},

           {"j,intra_parallel", "Thread Number Per-Instance",
            cxxopts::value<int>()->default_value("4")},

           {"pre_batch_size", "Preprocessor batch size",
            cxxopts::value<int>()->default_value("32")},

           {"b,batch_size", "Offline Model Batch Size",
            cxxopts::value<int>()->default_value("1")},

           {"split_len", "Sequence split len",
            cxxopts::value<int>()->default_value("-1")},

           {"enable_bf16", "Whether enable bf16 for prediction & joint",
            cxxopts::value<bool>()->default_value("false")},

           {"k,test_scenario", "Test scenario [Offline, Server]",
            cxxopts::value<std::string>()->default_value("Offline")},

           {"preprocessor", "Whether enbale audio preprocess or not",
            cxxopts::value<bool>()->default_value("false")},

           {"p,profiler", "Whether output trace json or not",
            cxxopts::value<bool>()->default_value("false")},

           {"f,profiler_folder",
            "If profiler is True, output json in profiler_folder",
            cxxopts::value<std::string>()->default_value("logs")},

           {"profiler_iter", "Profile iteration number",
            cxxopts::value<int>()->default_value("-1")},

           {"c,mlperf_config", "Configuration File for LoadGen",
            cxxopts::value<std::string>()->default_value("mlperf.conf")},

           {"u,user_config", "User Configuration for LoadGen",
            cxxopts::value<std::string>()->default_value("user.conf")},

           {"o,output_dir", "Test Output Directory",
            cxxopts::value<std::string>()->default_value("mlperf_output")},

           {"a,accuracy", "Run test in accuracy mode instead of performance",
            cxxopts::value<bool>()->default_value("false")}});

  auto parsed_opts = opts.parse(argc, argv);

  auto model_file = parsed_opts["model_file"].as<std::string>();
  auto sample_file = parsed_opts["sample_file"].as<std::string>();
  auto preprocessor_file = parsed_opts["preprocessor_file"].as<std::string>();
  auto pre_parallel = parsed_opts["pre_parallel"].as<int>();
  auto inter_parallel = parsed_opts["inter_parallel"].as<int>();
  auto intra_parallel = parsed_opts["intra_parallel"].as<int>();
  auto pre_batch_size = parsed_opts["pre_batch_size"].as<int>();
  auto batch_size = parsed_opts["batch_size"].as<int>();
  auto split_len = parsed_opts["split_len"].as<int>();
  auto enable_bf16 = parsed_opts["enable_bf16"].as<bool>();
  auto test_scenario = parsed_opts["test_scenario"].as<std::string>();
  auto preprocessor_flag = parsed_opts["preprocessor"].as<bool>();
  auto profiler_flag = parsed_opts["profiler"].as<bool>();
  auto profiler_folder = parsed_opts["profiler_folder"].as<std::string>();
  auto profiler_iter = parsed_opts["profiler_iter"].as<int>();
  auto mlperf_conf = parsed_opts["mlperf_config"].as<std::string>();
  auto user_conf = parsed_opts["user_config"].as<std::string>();
  auto output_dir = parsed_opts["output_dir"].as<std::string>();
  auto accuracy_mode = parsed_opts["accuracy"].as<bool>();

  mlperf::TestSettings testSettings;
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = output_dir;

  RNNTSUT sut(
    sample_file, model_file, preprocessor_file,
    pre_parallel, inter_parallel, intra_parallel,
    pre_batch_size, batch_size, split_len,
    enable_bf16, test_scenario, preprocessor_flag,
    profiler_flag, profiler_folder, profiler_iter);
  
  testSettings.scenario = scenario_map[test_scenario];
  testSettings.FromConfig(mlperf_conf, "rnnt", test_scenario);
  testSettings.FromConfig(user_conf, "rnnt", test_scenario);

  if (accuracy_mode)
    testSettings.mode = mlperf::TestMode::AccuracyOnly;

  std::cout << "Start " << test_scenario << " testing..." << std::endl;
  mlperf::StartTest(&sut, sut.GetQSL(), testSettings, logSettings);
  std::cout << "Testing done." << std::endl;

  return 0;
}
