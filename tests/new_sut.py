import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "../"))

import array
import mlperf_loadgen as lg
import multiprocessing as mp
import numpy as np
import threading
import toml

from datasets.preprocessing import AudioPreprocessing
from rnnt_qsl import RNNTQSL
from tqdm import tqdm
from utils import *


def get_num_cores():
    cmd = "lscpu | awk '/^Core\(s\) per socket:/ {cores=$NF}; /^Socket\(s\):/ {sockets=$NF}; END{print cores*sockets; print cores; print sockets}'"
    lscpu = os.popen(cmd).readlines()
    return int(str.rstrip(lscpu[0])), int(str.rstrip(lscpu[1])), int(str.rstrip(lscpu[2]))

def block_until(counter, num_ins, t):
    while counter.value < num_ins:
        time.sleep(t)


class Input(object):
    def __init__(self, id_list, idx_list):
        assert isinstance(id_list, list)
        assert isinstance(idx_list, list)
        assert len(id_list) == len(idx_list)
        self.query_id_list = id_list
        self.query_idx_list = idx_list

class EncodeOutput(object):
    def __init__(self, id_list, idx_list, logits, logits_lens, dur_fea, dur_enc):
        assert isinstance(id_list, list)
        assert isinstance(idx_list, list)
        assert len(id_list) == len(idx_list)
        self.query_id_list = id_list
        self.query_idx_list = idx_list
        self.logits = logits
        self.logits_lens = logits_lens
        self.dur_enc = dur_enc
        self.dur_fea = dur_fea

class Output(object):
    def __init__(self, query_id, transcript):
        self.query_id = query_id
        self.transcript = transcript

class AddNone(object):
    def __init__(self):
        self.value = 0


class InQueue():
    def __init__(self, input_queue_list, qsl, seq_cutoff_list,
                 batch_size_list):

        self.input_queue_list = input_queue_list
        self.qsl = qsl
        self.seq_cutoff_list = seq_cutoff_list
        self.num_queues = len(input_queue_list)
        self.batch_size_list = batch_size_list
        self.query_batcher = [[] for _ in range(self.num_queues)]
        # record the first arrival time of the query batch
        self.query_batcher_time = [None for _ in range(self.num_queues)]
        self.curr_query_count = 0

    def put(self, query_samples):

        if query_samples==None:
            # no more calls to put function
            # submit remaining queries in query batcher to input queues
            # process remaining queries with BS=1
            for i in range(self.num_queues):
                for q in self.query_batcher[i]:
                    input_item = Input([q.id], [q.index])
                    self.input_queue_list[i].put(input_item)
            return

        self.curr_query_count += len(query_samples)

        for sample in query_samples:
            for i in range(self.num_queues):
                idx = sample.index  #BS=1
                waveform = self.qsl[idx]
                if len(waveform) <= self.seq_cutoff_list[i]:
                    if self.query_batcher[i] == []:
                        self.query_batcher_time[i] = time.time()
                    self.query_batcher[i].append(sample)
                    # put queries in queue if BS treshold reached
                    if len(self.query_batcher[i]) == self.batch_size_list[i]:
                        qid_list, qidx_list = [], []
                        for q in self.query_batcher[i]:
                          qid_list.append(q.id)
                          qidx_list.append(q.index)
                        input_item = Input(qid_list, qidx_list)
                        self.input_queue_list[i].put(input_item)
                        self.query_batcher[i] = []
                        self.query_batcher_time[i] = None
                    break
        for i in range(self.num_queues):
            if self.query_batcher_time[i] != None and time.time() - self.query_batcher_time[i] > 0.1:
                #print ("issue sample in queue {} because time is pressing, samples in queue {}".format(i, len(self.query_batcher[i])))
                qid_list, qidx_list = [], []
                for q in self.query_batcher[i]:
                  qid_list.append(q.id)
                  qidx_list.append(q.index)
                input_item = Input(qid_list, qidx_list)
                self.input_queue_list[i].put(input_item)
                self.query_batcher[i] = []
                self.query_batcher_time[i] = None

class Consumer(mp.Process):
    def __init__(self, task_queue, task_queue_group, result_queue, lock, init_counter,
                 rank, core_list, qsl, config_toml, checkpoint_path, dataset_dir,
                 manifest_filepath, perf_count, profile, int8, bf16, warmup,
                 configure_path, mode, total_fea=0, total_enc=0, total_dec=0, verbose="0"):

        mp.Process.__init__(self)

        ### sub process
        self.task_queue = task_queue
        self.task_queue_group = task_queue_group
        self.result_queue = result_queue
        if (result_queue != None):
            self.result_queue.put(AddNone())
        self.lock = lock
        self.init_counter = init_counter
        self.rank = rank
        self.core_list = core_list
        self.num_cores = len(self.core_list)

        self.qsl = qsl
        self.config_toml = config_toml
        self.checkpoint_path = checkpoint_path
        self.dataset_dir = dataset_dir
        self.manifest_filepath = manifest_filepath
        self.perf_count = perf_count
        self.configure_path = configure_path
        self.queue_wait = 0.0
        self.queue_count = 0
        # by default, when get a none, the input queue is empty
        # however we can add 'none count' by add input with non-count
        self.wait_for_none_count = 1

    def init_model(self):
        # create preprocessor
        if args.enable_preprocess and os.path.exists(args.toml_path):
            config = toml.load(args.toml_path)
            featurizer_config = config["input_eval"]
            self.preprocessor = AudioPreprocessing(**featurizer_config).eval()
        else:
            self.preprocessor = None
        # create model
        if args.run_mode == "quant":
            from modeling_rnnt_quant import RNNT, GreedyDecoder
        else:
            from modeling_rnnt import RNNT, GreedyDecoder
        rnnt = RNNT(model_path, args.run_mode, args.load_jit, args.split_fc1).eval()
        self.model = GreedyDecoder(rnnt, args.load_jit)
        self.enable_preprocess = (self.preprocessor != None)
        self.load_jit = args.load_jit
        self.save_jit = args.save_jit
        self.batch_size = batch_size
        self.scenario = args.scenario if args.run_mode != "calib" else None
        if self.save_jit:
            if self.enable_preprocess:
                self.preprocessor = jit_module(self.preprocessor)
            self.model.rnnt = jit_model(self.model.rnnt)

    def inference(self, batch_idx):
        with torch.no_grad():
            if self.enable_preprocess:
                wavs = torch.nn.utils.rnn.pad_sequence(
                    [self.qsl[idx][0] for idx in batch_idx], batch_first=True)
                wav_lens = torch.tensor(
                    [self.qsl[idx][1] for idx in batch_idx])
                feas, fea_lens = self.preprocessor(wavs, wav_lens)
            else:
                feas = torch.nn.utils.rnn.pad_sequence(
                    [self.qsl[idx][0] for idx in batch_idx])
                fea_lens = torch.tensor(
                    [self.qsl[idx][1] for idx in batch_idx])
            results = self.model(feas, fea_lens)
        return results

    def run_queue(self):
        next_task = self.task_queue.get()
        if next_task is None:
            self.task_queue.task_done()
            self.wait_for_none_count -= 1
            if self.wait_for_none_count <= 0:
                self.result_queue.put(None)
                return False
            else:
                return True

        if isinstance(next_task, AddNone):
            self.wait_for_none_count += 1
            return True

        results = self.inference(next_task.query_idx_list)
        for id, res in zip(next_task.query_id_list, results):
            self.result_queue.put(Output(id, trans))
        self.task_queue.task_done()
        return True

    def run(self):
        str_core_list='{}'.format(self.core_list).replace(' ','').replace('[','').replace(']','')
        os.environ['OMP_NUM_THREADS'] = '{}'.format(self.num_cores)
        os.environ['KMP_AFFINITY'] = 'explicit,proclist=[{}]'.format(str_core_list)
        os.sched_setaffinity(self.pid, self.core_list)
        torch.set_num_threads(self.num_cores)
        torch.set_num_interop_threads(1)

        self.init_model()

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        while self.run_queue():
            pass

global t_start
def response_loadgen(out_queue):
    global finish_count
    global query_count
    global t_start
    out_queue_cnt = 0
    os.sched_setaffinity(os.getpid(), [0])
    while True:
        next_task = out_queue.get()
        if next_task is None:
            print("Exiting response thread")
            break

        if isinstance(next_task, AddNone):
            continue

        query_id = next_task.query_id
        transcript = next_task.transcript
        response_array = array.array('q', transcript)
        bi = response_array.buffer_info()
        response = lg.QuerySampleResponse(query_id, bi[0],
                                          bi[1] * response_array.itemsize)
        lg.QuerySamplesComplete([response])
        out_queue_cnt += 1
        finish_count += 1
        if debug:
            if finish_count == 1:
                t_start = time.time()
                print("Finish {} of {} samples".format(finish_count, query_count), end='\r')
            else:
                elapsed = time.time() - t_start
                rate = elapsed/(finish_count - 1)
                remaining_time = (query_count - finish_count)*rate
                print("Finish {} of {} samples, remaining {} seconds.".format(finish_count, query_count, int(remaining_time)), end='\r')

    print("Finish processing {} samples in this queue".format(out_queue_cnt))


class PytorchSUT:
    def __init__(self, model_path, dataset_dir, batch_size=1, args=None, **kwargs):
        self.num_cores, self.cores_per_socket, self.num_sockets = get_num_cores()
        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.output_queue = mp.Queue()
        self.input_queue = mp.JoinableQueue()
        self.decode_queue = mp.JoinableQueue()

        # server-specific
        self.num_queues = None
        self.core_count_list = []
        self.num_instance_list = []
        self.seq_cutoff_list = []
        self.batch_size_list = []
        self.run_mode= []
        self.input_queue_list = []

        # create queue list
        for _ in range(self.num_queues):
            self.input_queue_list.append(mp.JoinableQueue())

        # create qsl & sut
        self.qsl = RNNTQSL(dataset_dir)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        self.issue_queue = InQueue(self.input_queue_list, self.qsl,
                                   self.seq_cutoff_list, self.batch_size_list)

        ### worker process
        self.consumers = []
        self.decoders = []
        cur_core_idx = self.cores_for_loadgen
        rank = 0
        for i in range(self.decoder_num_instances):
            self.decoders.append(
                Consumer(self.decode_queue, -1, self.output_queue, self.lock, self.init_counter, rank,
                        [j for j in range(cur_core_idx, cur_core_idx+self.cores_for_decoder)],
                        self.qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath,
                        perf_count, profile, int8, bf16, warmup, configure_path, 'dec',
                        self.total_fea, self.total_enc, self.total_dec, verbose))
            rank += 1
            cur_core_idx += self.cores_for_decoder
        start_cores = [cur_core_idx]+[0]*(self.num_sockets-1)
        cur_socket = 0
        for i in range(self.num_queues-1, -1, -1):
            curr_cores_per_instance = self.core_count_list[i]
            for _ in range(self.num_instance_list[i]):
                while (start_cores[cur_socket] + curr_cores_per_instance > self.cores_per_socket):
                    cur_socket = (cur_socket+1) % self.num_sockets
                cur_core_idx = start_cores[cur_socket] + cur_socket*self.cores_per_socket
                #print ("assign instance from queue {} to core [{}:{}]".format(i, cur_core_idx, cur_core_idx+curr_cores_per_instance-1))
                self.consumers.append(
                    Consumer(self.input_queue_list[i], i,
                            self.decode_queue if self.run_mode[i]=='enc' else self.output_queue,
                            self.lock, self.init_counter, rank,
                            [i for i in range(cur_core_idx, cur_core_idx + curr_cores_per_instance)],
                            self.qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath,
                            perf_count, profile, int8, bf16, warmup, configure_path, self.run_mode[i],
                            self.total_fea, self.total_enc, self.total_dec, verbose))
                rank += 1
                start_cores[cur_socket] += curr_cores_per_instance
                cur_socket = (cur_socket+1) % self.num_sockets
        self.num_instances = len(self.consumers) + len(self.decoders)

        ### start worker process
        for d in self.decoders:
            d.start()
        for c in self.consumers:
            c.start()

        ### wait until all sub processes are ready
        block_until(self.init_counter, self.num_instances, 2)

        ### start response thread
        self.response_worker = threading.Thread(
            target=response_loadgen, args=(self.output_queue,))
        self.response_worker.daemon = True
        self.response_worker.start()

    def issue_queries(self, samples):
        global start_time
        global query_count
        if self.scenario == "Offline":
            samples.sort(key=lambda s: self.qsl[s.index][1].item(), reverse=True)
        self.issue_queue.put(samples)
        query_count += len(samples)

    def flush_queries(self):
        self.issue_queue.put(None)

    def __del__(self):
        ### clear up sub processes
        for i in range(self.num_queues):
            self.input_queue_list[i].join()
            for _ in range(self.num_instance_list[i]):
                self.input_queue_list[i].put(None)

        for c in self.consumers:
            c.join()

        for i in range(len(self.decoders)):
            self.decode_queue.put(None)
        for d in self.decoders:
            d.join()

        self.output_queue.put(None)
        self.cal_split_latencies()
        print("Finished destroying SUT.")

