# -*- coding: utf-8 -*-

import copy
import math
import pickle
import random
import traceback
# Ahmed - account for label occurances
from collections import Counter

import numpy as np
import torch
# Ahmed imported modules
import wandb

from fl_aggregator_libs import *
from resource_manager import ResourceManager
from utils.utils_model import cosine_sim, kl_divergence, normalize

#import pandas as pd

class Aggregator(object):
    """This centralized aggregator collects training/testing feedbacks from executors"""
    def __init__(self, args):
        #print(f'Starting Aggregator {args}')
        logging.info(f"Job args {args}")

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')
        self.executors = [int(v) for v in str(args.learners).split('-')]
        self.num_executors = len(self.executors)

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.virtual_client_clock = {}
        self.round_duration = 0.
        self.resource_manager = ResourceManager()
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model = None

        # list of parameters in model.parameters()
        self.model_in_update = []
        self.last_global_model = []

        # ======== channels ========
        self.server_event_queue = {}
        self.client_event_queue = Queue()
        self.control_manager = None
        # event queue of its own functions
        self.event_queue = collections.deque()

        # ======== runtime information ========
        self.tasks_round = 0
        self.sampled_participants = []

        self.round_stragglers = []
        self.model_update_size = 0.

        self.collate_fn = None
        self.task = args.task
        self.epoch = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # number of registered executors
        self.registered_executor_info = 0
        self.test_result_accumulator = []
        self.testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                        'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()}

        self.gradient_controller = None
        if self.args.gradient_policy == 'yogi':
            from utils.yogi import YoGi
            self.gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)

        #Ahmed - metrics, stale and availibility variables
        self.clients_select_count = {}
        self.clients_success_count = {}
        self.clients_fail_count = {}
        # Ahmed - define the stale updates list
        self.staleWeights = {}
        self.staleRemainDuration = {}
        self.stale_rounds = {}
        self.round_stale_param = {}
        self.update_stale_rounds = {}
        self.round_stale_updates = 0
        self.straggler_comp_time = {}
        self.strugglers_to_run = []
        self.dropout_clients = []

        self.last_round_stragglers = []
        self.last_round_duration = 0

        self.round_failures = 0
        self.mov_avg_deadline = 0
        self.deadline = 0 if (args.exp_type == 1 or args.exp_type == 3) else args.deadline
        self.attended_clients = 0
        self.unique_attend_clients = []
        self.acc_accumulator = []
        self.acc_5_accumulator = []
        self.train_accumulator = []
        self.completion_accumulator = []

        self.total_compute = 0
        self.total_communicate = 0

        self.total_updates = 0
        self.unused_stale = 0
        self.round_update = False

        self.sum_update_norm = []
        self.count_param_updates = []
        self.new_deltas = []

        self.update_clients = []

        self.param_update_count = 0

        self.label_counter = Counter()
        self.no_straggler_rounds = 0

        self.last_model_diff = 0

        #Ahmed - dynamically set the per round total workers
        self.total_worker = args.total_worker if args.total_worker > 0 else args.initial_total_worker

        # Ahmed - used for choosing a replacement client among online if it matches the stale one
        self.rng = None #intialized after setting the seed

        self.wandb_run = None

        self.priority_clients = []
        self.stale_computation = {}
        self.stale_communication = {}
        self.new_compute = 0
        self.new_communicate = 0
        self.stale_compute = 0
        self.stale_communicate = 0

        self.time_window = 0

        self.bad_update = 0
        self.good_update = 0
        self.bad_param = 0
        self.good_param = 0

        self.importance_list = []
        self.client_importance = {}
        self.client_samples = {}

        self.new_clients = {}
        self.stale_clients = {}

        self.last_run_round = {}

        self.client_ratio = {}
        self.scale_coff = args.scale_coff if args.scale_coff > 0 else 10.0

        # ======== Task specific ============
        self.imdb = None           # object detection

    def setup_env(self):
        self.setup_seed(seed=self.this_rank)

        # set up device
        if self.args.use_cuda:
            if self.device == None:
                for i in range(torch.cuda.device_count()):
                    try:
                        self.device = torch.device('cuda:'+str(i))
                        torch.cuda.set_device(i)
                        _ = torch.rand(1).to(device=self.device)
                        logging.info(f'End up with cuda device ({self.device})')
                        break
                    except Exception as e:
                        assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'
            else:
                torch.cuda.set_device(self.device)
        logging.info(f'=== PS cuda device is preset to ({self.device})')

        #Ahmed - setup the seed again
        self.setup_seed(seed=self.this_rank)

        self.init_control_communication(self.args.ps_ip, self.args.manager_port, self.executors)
        self.init_data_communication()

    # Ahmed - for reporducbility - it does not work
    # Ahmed - https://github.com/pytorch/pytorch/issues/7068
    def setup_seed(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        #torch.use_deterministic_algorithms(True)

        # Ahmed - used for choosing a replacement client among online if it matches the stale one
        self.rng = random.Random()

    def init_control_communication(self, ps_ip, ps_port, executors):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Start to initiate {ps_ip}:{ps_port} for control plane communication ...")

        dummy_que = {executorId:Queue() for executorId in executors}
        # create multiple queue for each aggregator_executor pair
        for executorId in executors:
            BaseManager.register('get_server_event_que'+str(executorId), callable=lambda: dummy_que[executorId])

        dummy_client_que = Queue()
        BaseManager.register('get_client_event', callable=lambda: dummy_client_que)

        self.control_manager = BaseManager(address=(ps_ip, ps_port), authkey=b'FLPerf')
        self.control_manager.start()

        #self.server_event_queue = torch.multiprocessing.Manager().dict()
        for executorId in self.executors:
            self.server_event_queue[executorId] = eval('self.control_manager.get_server_event_que'+str(executorId)+'()')

        self.client_event_queue = self.control_manager.get_client_event()

    def init_data_communication(self):
        dist.init_process_group(self.args.backend, rank=self.this_rank, world_size=len(self.executors) + 1)


    def init_model(self):
        """Load model"""
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb("voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

        return init_model()

    def init_client_manager(self, args):
        """
            Currently we implement two client managers:
            1. Random client sampler
                - it selects participants randomly in each round
                - [Ref]: https://arxiv.org/abs/1902.01046
            2. Kuiper sampler
                - Kuiper prioritizes the use of those clients who have both data that offers the greatest utility
                  in improving model accuracy and the capability to run training quickly.
                - [Ref]: https://arxiv.org/abs/2010.06081
        """

        # sample_mode: random or kuiper
        client_manager = clientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        # load client profiles
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def executor_info_handler(self, executorId, info):

        self.registered_executor_info += 1

        # have collected all executors
        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout

        if self.registered_executor_info == self.num_executors:

            clientId = 1

            for index, _size in enumerate(info['size']):

                # since the worker rankId starts from 1, we also configure the initial dataId as 1
                mapped_id = clientId%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
                systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})

                self.client_manager.registerClient(executorId, clientId, size=_size, speed=systemProfile)
                self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                    upload_epoch=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)

                # Ahmed - initiate the client selection and run metrics
                self.clients_select_count[clientId] = 0
                self.clients_success_count[clientId] = 0
                self.clients_fail_count[clientId] = 0

                #Ahmed - record the last run round
                self.last_run_round [clientId] = 0

                clientId += 1
                #logging.info(f'SETUP client {clientId} samples: {_size} batchsize:{self.args.batch_size}')

            #Ahmed - update the durations of the clients and their execution cost
            self.client_manager.updateClientCosts()

            #Ahmed - add this information to the config
            info = self.client_manager.getDataInfo()
            for key in info:
                self.wandb_run.config[key] = info[key]
            logging.info("Info of all feasible clients {}".format(info))

            # start to sample clients
            self.round_completion_handler()


    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """We try to remove dummy events as much as possible, by removing the stragglers/offline clients in overcommitment"""

        sampledClientsReal = []
        dropoutClients = {}
        completionTimes = []
        completed_client_clock = {}
        client_costs = {}

        #Ahmed - choose random set of clients to drop out based on dropout_ratio
        sampled_dropouts = []
        if args.dropout_ratio > 0:
            sampled_dropouts = self.rng.sample(sampled_clients, int(len(sampled_clients) * args.dropout_ratio))
        elif args.exp_type == 3 and not args.random_behv:
            val = self.rng.uniform(args.overcommitment * 0.5, args.overcommitment * 1.5)
            sampled_dropouts = self.rng.sample(sampled_clients, int(len(sampled_clients) * val))

        # 1. remove dummy clients that are not available to the end of training
        for client_to_run in sampled_clients:
            #logging.info(f'{client_to_run} {self.client_conf}')
            client_cfg = self.client_conf.get(client_to_run, self.args)

            #Ahmed - update to obtain the cost from the clients_exe_cost dict of client_manager
            exe_cost = self.client_manager.getCompletionTime(client_to_run)
            roundDuration = exe_cost['computation'] + exe_cost['communication']
            client_costs[client_to_run] = exe_cost

            #Ahmed - change to dropout the client if he is not online at end of training or he is a dropout candidate
            if self.client_manager.isClientActive(client_to_run, self.global_virtual_clock, roundDuration) and client_to_run not in sampled_dropouts:

            #Ahmed - change to use the full period instead of checking the end of the period
            #if self.client_manager.isAvailable(client_to_run, self.global_virtual_clock, roundDuration, 1) and client_to_run not in sampled_dropouts:
                sampledClientsReal.append(client_to_run)
                completionTimes.append(roundDuration)
                completed_client_clock[client_to_run] = exe_cost

                # Ahmed - account for execution time and communication for online clients
                self.total_compute += exe_cost['computation']
                self.total_communicate += exe_cost['communication']
            else:
                if self.time_window >= roundDuration and client_to_run in self.priority_clients:
                    logging.info(f'====== Pruning WRONG PRIORITY DROPOUT: client {client_to_run}, window {self.time_window} duration {roundDuration}')
                dropoutClients[client_to_run] = roundDuration
                # Ahmed - account for execution time for dropout clients as they will not communicate
                self.total_compute += exe_cost['computation']

        #Ahmed - if we do not filter dropouts, we inflate their completion time
        if args.no_filter_dropouts > 0:
            for client_to_run in dropoutClients:
                comp_time = args.no_filter_dropouts * max(completionTimes)  # exe_cost
                sampledClientsReal.append(client_to_run)
                completionTimes.append(comp_time)
                completed_client_clock[client_to_run] = client_costs[client_to_run]

        num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))

        # 2. get the top-k completions to remove stragglers
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])

        # Ahmed - Change to only top 80% of the clients to simulate failure of 20% with deadline
        if args.deadline_percent > 0:
            num_clients_to_collect = int(math.floor(args.deadline_percent * num_clients_to_collect))

        # Ahmed - apply fixed deadline or moving average if it is not zero
        # Ahmed - apply deadline for SAFA exp_type=0
        if args.exp_type == 0 or args.exp_type == 2: # or args.exp_type==4:
            deadline = args.deadline
            if args.deadline <= 0:
                if self.mov_avg_deadline > 0:
                    deadline = self.mov_avg_deadline
                else:
                    deadline = self.args.initial_deadline
            if deadline > 0:
                for index, i in enumerate(sortedWorkersByCompletion):
                    if completionTimes[i] > deadline:
                        num_clients_to_collect = index
                        break
            logging.info("====Apply deadline {}:{}:{} or percent {}, before num {} final num {} sorted workers {} durations: {}".format(
                    deadline, args.target_ratio, deadline, args.deadline_percent, len(sortedWorkersByCompletion),
                    num_clients_to_collect, sortedWorkersByCompletion, completed_client_clock))

        #Ahmed - move on to next round and put all clients except for fastest one into  stale cache
        if args.stale_all:
            num_clients_to_collect = 1

        top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]
        dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:] if
                         sampledClientsReal[k] not in dropoutClients]
        round_duration =  completionTimes[top_k_index[-1]] if len(top_k_index) else 0.0

        return clients_to_run, dummy_clients, dropoutClients, completed_client_clock, round_duration, completionTimes, sampled_dropouts

    def run(self):
        try:
            self.setup_env()
            self.model = self.init_model()

            #Ahmed - get the param count
            self.model_param_count = 0
            for idx, param in enumerate(self.model.parameters()):
                self.model_param_count += 1
                #self.sum_update_norm.append(0.0)
                self.count_param_updates.append(0)

            self.save_last_param()

            self.model_update_size = sys.getsizeof(pickle.dumps(self.model))/1024.0*8. # kbits
            self.client_profiles = self.load_client_profile(file_path=self.args.device_conf_file)

            #Ahmed - start wandb run and its communication
            self.start_wandb()

            self.start_event()
            self.event_monitor()
        except Exception as e:
            traceback.print_exc()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # Ahmed - add print of the stack call trace to give meaningful debugging information
            logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
            print('Aggregator Exception - sending stop signal and terminating the process group Now!')
            self.broadcast_msg('emergency_stop')
            time.sleep(5)
            self.stop()
            exit(-1)

    def start_event(self):
        #self.model_in_update = [param.data for idx, param in enumerate(self.model.parameters())]
        #self.event_queue.append('report_executor_info')
        pass

    def broadcast_msg(self, msg):
        for executorId in self.executors:
            self.server_event_queue[executorId].put_nowait(msg)


    def broadcast_models(self):
        """Push the latest model to executors"""
        # self.model = self.model.to(device='cpu')

        # waiting_list = []
        for param in self.model.parameters():
            temp_tensor = param.data.to(device='cpu')
            for executorId in self.executors:
                dist.send(tensor=temp_tensor, dst=executorId)
                # req = dist.isend(tensor=param.data, dst=executorId)
                # waiting_list.append(req)

        # for req in waiting_list:
        #     req.wait()

        # self.model = self.model.to(device=self.device)


    def select_participants(self, select_num_participants, time_window=0):
        #Ahmed - change to include the deadline for availability calculations
        #return sorted(self.client_manager.resampleClients(int(select_num_participants*overcommitment), cur_time=self.global_virtual_clock))
        # Ahmed - handle the case when the requested number is zero
        if select_num_participants <= 0:
            return [], []
        picked_clients, priority_clients = self.client_manager.resampleClients(select_num_participants, cur_time=self.global_virtual_clock, time_window=time_window)
        return sorted(picked_clients), sorted(priority_clients)


    def client_completion_handler(self, results): #, importance=1.0):
        """We may need to keep all updates from clients, if so, we need to append results to the cache"""

        clientId=results['clientId']

        complete_time = 0
        if clientId in self.virtual_client_clock:
            complete_time = self.virtual_client_clock[clientId]['computation']+self.virtual_client_clock[clientId]['communication']

        #Ahmed - record the number of attended and unique clients
        if clientId not in self.unique_attend_clients:
            self.unique_attend_clients.append(clientId)
        self.attended_clients += 1

        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': epoch_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}
        self.client_training_results.append(results)

        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        #Ahmed - accumlate train_acc and train_acc_5
        self.acc_accumulator.append(results['train_acc'] * 100)
        self.acc_5_accumulator.append(results['train_acc_5'] * 100)
        self.train_accumulator.append(results['train_loss'])
        self.completion_accumulator.append(complete_time)

        self.client_samples[clientId] = results['sample_count']

        #Ahmed - aggregate the label counters from the clients
        self.label_counter.update(results['label_counter'])

        #Ahmed - Handle and keep a copy of the stale client update
        isStale = True if (clientId in self.staleWeights and len(self.staleWeights[clientId]) == 0) else False
        if isStale:
            self.staleWeights[clientId] = copy.deepcopy(results['update_weight'])
            #complete_time = self.straggler_comp_time[clientId] * args.straggler_penalty
            logging.info('======== Aggregator received straggler client: id {} comp_time {} param_len {} stale {}'.format(clientId, complete_time, len(self.staleWeights[clientId]), len(self.staleWeights)))
        #else:
        self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
                    time_stamp=self.epoch, duration=complete_time)

        #Ahmed - only update the model if the round did not fail (we have enough new+stale clients)
        if self.round_update and not isStale and not args.send_delta:
            if len(self.model_in_update) == 0:
                self.model_in_update = [True]
                for idx, param in enumerate(self.model.parameters()):
                    param.data = torch.from_numpy(results['update_weight'][idx]).to(device=self.device) * self.client_importance[clientId]
                    self.count_param_updates[idx] += 1
            else:
                for idx, param in enumerate(self.model.parameters()):
                    param.data += torch.from_numpy(results['update_weight'][idx]).to(device=self.device) * self.client_importance[clientId]
                    self.count_param_updates[idx] += 1
            self.update_clients.append(clientId)

            #Ahmed - delete the update to free memory
            del results['update_weight']
            self.param_update_count += 1

        # Ahmed - invoke the garabage collector to free memory
        gc.collect()

    def save_last_param(self):
        #Ahmed - maintain the state of the last global model and reset sum of state norms
        self.last_global_model = [param.data.clone() for param in self.model.parameters()]

    def round_weight_handler(self, last_model, current_model):
        if self.epoch > 1:
            if self.args.gradient_policy == 'yogi':
                last_model = [x.to(device=self.device) for x in last_model]
                current_model = [x.to(device=self.device) for x in current_model]

                diff_weight = self.gradient_controller.update([pb-pa for pa, pb in zip(last_model, current_model)])

                for idx, param in enumerate(self.model.parameters()):
                    param.data = last_model[idx] + diff_weight[idx]

            elif self.args.gradient_policy == 'qfedavg':
                learning_rate, qfedq = self.args.learning_rate, self.args.qfed_q
                Deltas, hs = None, 0.
                last_model = [x.to(device=self.device) for x in last_model]

                for result in self.client_training_results:
                    # plug in the weight updates into the gradient
                    grads = [(u - torch.from_numpy(v).to(device=self.device)) * 1.0 / learning_rate for u, v in zip(last_model, result['update_weight'])]
                    loss = result['moving_loss']

                    if Deltas is None:
                        Deltas = [np.float_power(loss+1e-10, qfedq) * grad for grad in grads]
                    else:
                        for idx in range(len(Deltas)):
                            Deltas[idx] += np.float_power(loss+1e-10, qfedq) * grads[idx]

                    # estimation of the local Lipchitz constant
                    hs += (qfedq * np.float_power(loss+1e-10, (qfedq-1)) * torch.sum(torch.stack([torch.square(grad).sum() for grad in grads])) + (1.0/learning_rate) * np.float_power(loss+1e-10, qfedq))

                # update global model
                for idx, param in enumerate(self.model.parameters()):
                    param.data = last_model[idx] - Deltas[idx]/(hs+1e-10)
    def advance_round(self):
        # update the virtual clock
        # Ahmed - change to use the deadline if set
        duration = self.round_duration
        if args.exp_type == 0 or args.exp_type == 2:  # if self.deadline != 0:
            duration = self.deadline
        self.global_virtual_clock += duration

        self.epoch += 1

    def round_completion_handler(self, importance=1.0):
        self.advance_round()

        if self.epoch % self.args.decay_epoch == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        #Ahmed - change to apply weight updates if round not failed
        if self.round_update:
            self.round_weight_handler(self.last_global_model, [param.data.clone() for param in self.model.parameters()])

            #Ahmed - Apply the stale updates after the weight handler - to only apply YoGI/Prox to non stale updates only
            #Ahmed - this did not work out as it leads to divergent behaviour
            #self.stale_clients_handler(importance=importance)

        # TODO: Ahmed - this is might be wrong, does not match the paper, in the paper, page 6, the straggler (not ran) clients are not rewarded
        avgUtilLastEpoch = sum(self.stats_util_accumulator)/max(1, len(self.stats_util_accumulator))
        if args.stale_update == 0:
            # assign avg reward to explored, but not ran workers
            for clientId in self.round_stragglers:
                complete_time=self.virtual_client_clock[clientId]['computation']+self.virtual_client_clock[clientId]['communication']
                self.client_manager.registerScore(clientId, avgUtilLastEpoch, time_stamp=self.epoch, duration= complete_time,success=False)

        avg_loss = sum(self.loss_accumulator)/max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, Epoch: {self.epoch}, Planned participants: " + \
            f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # update select participants
        # Ahmed - use the version with availability prioritization
        overcommitment = 1.0
        # Ahmed - original oort setting
        if args.exp_type == 1 or args.exp_type == 3:
            overcommitment = self.args.overcommitment

        if (self.args.deadline <= 0) or (args.exp_type == 1 or args.exp_type == 3):
            if self.mov_avg_deadline == 0:
                # Ahmed - set the initial deadline to something large
                self.deadline = self.args.initial_deadline
            else:
                self.deadline = self.round_duration

        self.mov_avg_deadline = (1 - args.deadline_alpha) * self.deadline + args.deadline_alpha * self.mov_avg_deadline
        self.time_window = self.deadline if self.args.deadline > 0 else self.mov_avg_deadline

        num_to_sample = round(self.total_worker * overcommitment)
        # Ahmed - adapt the number of selected clients but not for SAFA
        if not str(args.wandb_tags).startswith('safa'): #and self.tasks_round > 0:
            if self.args.adapt_selection == 1:
                stale_ones, unused_stale = self.get_stale_status(self.mov_avg_deadline)
                if stale_ones > 0:
                    if self.args.adapt_selection_cap > 0:
                        num_to_sample = max(self.args.adapt_selection_cap * num_to_sample, num_to_sample - stale_ones)
                    else:
                        num_to_sample = max(1, num_to_sample - stale_ones)
            elif self.args.adapt_selection == 2:
                stale_ones, unused_stale = self.get_stale_status(self.mov_avg_deadline)
                if stale_ones > 0:
                    num_to_sample = 0

        #Ahmed - SAFA uses all online client for training and the # of workers to wait for is the target ratio
        if str(args.wandb_tags).startswith('safa'): #args.exp_type == 4:
            self.sampled_participants = self.client_manager.getFeasibleClients(self.global_virtual_clock)
            self.total_worker = int(len(self.sampled_participants) * self.args.target_ratio)
        else:
            self.sampled_participants, self.priority_clients = self.select_participants(select_num_participants=num_to_sample, time_window=self.time_window)

        #Ahmed - count the number of times a client is selected
        for c in self.sampled_participants:
            self.clients_select_count[c] += 1

        #Ahmed - changed the way clients are filtered using deadline or deadline_percent
        clientsToRun, round_stragglers, dropout_clients, virtual_client_clock, round_duration, client_completions, sampled_dropouts = self.tictak_client_tasks(self.sampled_participants, self.total_worker)

        #Ahmed - update the round duration, ensure it is updated before updating the deadline for next round
        self.last_round_duration = self.round_duration
        self.round_duration = round_duration

        #Ahmed - update the deadline based on the new stragglers
        if (args.exp_type == 0 or args.exp_type == 2):
            #Ahmed - round duration should be the deadline if not last_worker
            if not args.last_worker or self.round_duration <= 0:
                self.round_duration = self.mov_avg_deadline #self.deadline
            #Ahmed - update the deadline
            if self.deadline > 0:
                ratio = 0
                self.deadline = args.deadline

        # Ahmed - filter out the clients with pending stale updates if they were selected again
        if self.args.stale_update != 0 or self.args.prohibit_reselect != 0:
            count = 0
            re_sel_client = []
            replacement_clients = []
            #Ahmed - reselect clients that are stragglers and did not submit an update yet
            for c in self.staleWeights:
                if c in clientsToRun:
                    count += 1
                    re_sel_client.append(c)
                    clientsToRun.remove(c)
                    logging.info(f'====== Aggreagator REMOVE: straggler client {c}')

            #Ahmed - reselect clients that were ran less prohibit_select rounds
            if self.args.prohibit_reselect and self.args.avail_priority:
                for c in clientsToRun:
                    if self.last_run_round[c] != 0 and self.epoch - self.last_run_round[c] < self.args.prohibit_reselect:
                        count += 1
                        re_sel_client.append(c)
                        clientsToRun.remove(c)
                        logging.info(f'====== Aggreagator REMOVE: recently selected client {c}')

            re_sel_temp = re_sel_client.copy()
            while len(re_sel_client) > 0:
                c = re_sel_client.pop()
                #online_clients, _ = self.client_manager.getOnlineClients(cur_time=self.global_virtual_clock)
                index = self.rng.randint(0, len(self.client_manager.cur_online_clients) - 1)
                cid = self.client_manager.cur_online_clients[index]
                trials = 10
                while cid == c or cid in clientsToRun or cid in self.staleWeights:
                    index = self.rng.randint(0, len(self.client_manager.cur_online_clients) - 1)
                    cid = self.client_manager.cur_online_clients[index]
                    trials -= 1
                if trials >= 0:
                    replacement_clients.append(cid)
                    clientsToRun.append(cid)
                    #virtual_client_clock[cid] = self.client_manager.getCompletionTime(cid, batch_size=args.batch_size, upload_epoch=args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
                    virtual_client_clock[cid] = self.client_manager.getCompletionTime(cid)

            logging.info('Resample clients: round {} num {} sampled {} replace_clients {}:{} replacement_clients {} Run {} struggle {} clock: {}:{}'.format(\
                         self.epoch, self.total_worker, self.sampled_participants, count, re_sel_temp, replacement_clients, clientsToRun, round_stragglers, len(virtual_client_clock), virtual_client_clock))

        #Ahmed - ensure unqiue clients in clientstorun
        clientsToRun = list(set(clientsToRun))

        #Ahmed - record for the last round a client is selected
        for c in clientsToRun:
            self.last_run_round[c] = self.epoch

        # Ahmed - round's successful clients
        self.tasks_round = len(clientsToRun)

        if self.tasks_round > 0:
            # Ahmed - get the number of stale clients finishing this round
            self.round_stale_param, self.update_stale_rounds, self.round_stale_updates, self.unused_stale = self.get_stale_updates(self.round_duration)
        else:
            self.round_stale_param, self.update_stale_rounds, self.round_stale_updates, self.unused_stale = {}, {}, 0, 0

        #Ahmed - set the importance of each client finishing in this round
        del self.client_importance
        self.client_importance = {}
        for c in clientsToRun:
            self.client_importance[c] = 1.0
        for c in self.round_stale_param:
            self.client_importance[c] = 1.0

        for c in clientsToRun:
            self.clients_success_count[c] += 1
            self.new_compute += virtual_client_clock[c]['computation']
            self.new_communicate += virtual_client_clock[c]['communication']
        for c in round_stragglers:
            self.clients_fail_count[c] += 1

        #Ahmed - total updates is the new clients plus the stale ones finishing in this round
        self.total_updates = self.tasks_round + self.round_stale_updates

        #Ahmed - account for drop out clients if no_filter_dropouts is set
        if self.args.no_filter_dropouts:
            self.total_updates -= len(dropout_clients)

        # Ahmed - Perform round update only if the target ratio is met or target ratio is 0 (accept all)
        self.round_update = True

        #Ahmed - in SAFA if the new clients go to round stragglers if they do not meet the target number
        if str(args.wandb_tags).startswith('safa') and not self.round_update and args.stale_update != 0:
            round_stragglers.extend(clientsToRun)
            clientsToRun = []

        # Ahmed - extend the clients to run with the strugglers (stale clients) if they are not in the stale state
        # Ahmed - include stale clients if the model did not get enough boost last round
        run_stale_clients = True
        if self.args.model_boost:
            val1 = 0
            val2 = 0
            for idx, param in enumerate(self.model.parameters()):
                val1 += torch.norm(self.last_global_model[idx]) ** 2
                val2 += torch.norm(param.data) ** 2
            diff = abs(1.0 * (val2 - val1))
            ratio = abs(1.0 - (diff / self.last_model_diff))
            if self.last_model_diff > 0 and ratio >= 0.25:
                run_stale_clients = False
                logging.info(f'===== SKIP stragglers round {self.epoch}, val1 {val1} val2 {val2} diff {diff} last_diff {self.last_model_diff} decisions {ratio}')
            else:
                logging.info(f'===== TAKE stragglers round {self.epoch}, val1 {val1} val2 {val2} diff {diff} last_diff {self.last_model_diff} decisions {ratio}')
            self.last_model_diff = diff

        self.strugglers_to_run = []
        if run_stale_clients and args.stale_update != 0:
            for clientId in round_stragglers:
                if clientId not in self.staleWeights:
                    self.staleWeights[clientId] = []
                    self.stale_rounds[clientId] = 0
                    complete_time = virtual_client_clock[clientId]['computation'] + virtual_client_clock[clientId]['communication']
                    self.staleRemainDuration[clientId] = complete_time
                    self.straggler_comp_time[clientId] = complete_time
                    self.strugglers_to_run.append(clientId)
            clientsToRun.extend(self.strugglers_to_run)

        #Ahmed - account for compute of straggling clients
        for c in self.strugglers_to_run:
            self.stale_computation[c] = virtual_client_clock[c]['computation']
            self.stale_communication[c] = virtual_client_clock[c]['communication']

        logging.info(f"RUN Round:{self.epoch} duration:{round_duration} updates:{self.round_update}:{self.total_updates} run:{len(clientsToRun)}:{self.tasks_round}:{clientsToRun}\
         struggle:{len(self.strugglers_to_run)}:{len(round_stragglers)}:{round_stragglers} dropout:{len(dropout_clients)}:{dropout_clients}\
          clock:{len(virtual_client_clock)}:\n{virtual_client_clock}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.save_last_param()

        #Collect info from last round
        self.last_round_stragglers = self.round_stragglers

        self.round_stragglers = round_stragglers
        self.dropout_clients = dropout_clients
        self.virtual_client_clock = virtual_client_clock

        if self.epoch >= self.args.epochs:
            self.event_queue.append('stop')

        elif len(clientsToRun) <= 0:
            #Ahmed - handle the case when we have 0 clients to run
            self.event_queue.append('skip_round')

        elif self.epoch % self.args.eval_interval == 0:
            self.event_queue.append('update_model')
            self.event_queue.append('test')
        else:
            self.event_queue.append('update_model')
            self.event_queue.append('start_round')

    #Ahmed - define function resetting round metrics
    def round_reset_metrics(self):
        self.model_in_update = []
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        # Ahmed - reset accumulator metrics
        self.acc_5_accumulator = []
        self.acc_accumulator = []
        self.train_accumulator = []
        self.completion_accumulator = []

        self.unused_stale = 0
        self.param_update_count = 0
        self.importance_list = []

        self.sum_update_norm = []
        self.update_clients = []

        #self.client_importance = {}
        self.client_ratio = {}
        self.label_counter = Counter()

        #Ahmed - reset the param counter
        self.count_param_updates = []
        for idx, param in enumerate(self.model.parameters()):
            self.count_param_updates.append(0)

        gc.collect()

    def testing_completion_handler(self, results):
        self.test_result_accumulator.append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator) == len(self.executors):
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                    }
            else:
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'test_len': accumulator['test_len']
                    }

            if self.args.adapt_scale_coff:
                test_loss = accumulator['test_loss'] / accumulator['test_len']
                if self.last_test_loss is None:
                    self.last_test_loss = test_loss
                else:
                    ratio = (test_loss - self.last_test_loss) / self.last_test_loss
                    if  ratio > 0.1:
                        self.scale_coff = max(1, self.scale_coff / 1.5)
                    elif ratio < -0.1:
                        self.scale_coff = min(15, self.scale_coff * 1.5)

            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                    .format(self.epoch, self.global_virtual_clock, self.testing_history['perf'][self.epoch]['top_1'],
                    self.testing_history['perf'][self.epoch]['top_5'], self.testing_history['perf'][self.epoch]['loss'],
                    self.testing_history['perf'][self.epoch]['test_len']))

            #Ahmed - wandb logs
            if args.use_wandb :
                # Ahmed - log the test performance
                wandb.log({'Test/acc_top_1': self.testing_history['perf'][self.epoch]['top_1'],
                           'Test/acc_top_5': self.testing_history['perf'][self.epoch]['top_5'],
                           'Test/loss': self.testing_history['perf'][self.epoch]['loss'],
                           #'Test/data_len': self.testing_history['perf'][self.epoch]['test_len'],
                           }, step=self.epoch)

                #Ahmed - add more metrics on top 5 test accuracy
                if 'test' in self.testing_history['perf'][self.epoch]:
                    test_accs = np.asarray(self.testing_history['perf'][self.epoch]['test'])
                    num_test_clients = len(test_accs)
                    wandb.log({"Test/top_5_avg": np.average(test_accs, axis=0),
                               "Test/top_5_10p": np.percentile(test_accs, 10, axis=0),
                               "Test/top_5_50p": np.percentile(test_accs, 50, axis=0),
                               "Test/top_5_90p": np.percentile(test_accs, 90, axis=0),
                               "Test/top_5_var": np.var(test_accs),
                               "Round/test_clients": num_test_clients,
                               }, step=self.epoch)

                    if  num_test_clients > 0:
                        wandb.log({"Clients/test_top_5": wandb.Histogram(np_histogram=np.histogram(test_accs, bins=10)),}, step=self.epoch)

                        # Ahmed - Fairness of test accuracy
                        wandb.log({"Fairness/jain_top_5": (1.0 / num_test_clients * (np.sum(test_accs) ** 2) / np.sum(test_accs ** 2)),
                                   "Fairness/qoe_top_5": (1.0 - (2.0 * test_accs.std() / (test_accs.max() - test_accs.min()))),
                                   }, step=self.epoch)

                        # Compute an log cosine similarity metric with input vector a of
                        # clients' accuracies and b the same-length vector of 1s
                        vectors_of_ones = np.ones(num_test_clients)
                        wandb.log({"Fairness/cs_test_top_5":cosine_sim(test_accs, vectors_of_ones)}, step=self.epoch)

                        # Compute an log KL Divergence metric with input vector a of
                        # clients' normalized accuracies and the vector b of same-length
                        # generated from the uniform distribution
                        uniform_vector = np.random.uniform(0, 1, num_test_clients)
                        wandb.log({"Fairness/kl_test_top_5":kl_divergence(normalize(test_accs), uniform_vector)
                        }, step=self.epoch)

            self.event_queue.append('start_round')

    # Ahmed - handle clients deltas
    def client_deltas_handler(self):
        # if self.args.gradient_policy == 'yogi':
        #     self.new_deltas = self.gradient_controller.update(self.new_deltas)
        for result in self.client_training_results:
            clientId = result['clientId']
            for idx, param in enumerate(self.model.parameters()):
                temp_param = torch.from_numpy(result['update_weight'][idx]).to(device=self.device) * self.client_importance[clientId]
                param.data += temp_param

    #Ahmed - define stale clients handlers
    def get_stale_status(self, duration=0):
        stale_count = 0
        unused_stale = 0
        if len(self.staleWeights) > 0:
            for clientId in list(self.staleWeights):
                if (len(self.staleWeights[clientId]) > 0 and clientId in self.staleRemainDuration):
                    if self.staleRemainDuration[clientId] > 0 and self.staleRemainDuration[clientId] <= duration and self.stale_rounds[clientId] > 0:
                        if self.stale_rounds[clientId] <= args.stale_update or args.stale_update < 0:
                            stale_count += 1
                        else:
                            unused_stale += 1
        return stale_count, unused_stale

    def get_stale_updates(self, duration=0):
        round_stale_params = {}
        update_stale_rounds = {}
        stale_count = 0
        unused_stale = 0
        if len(self.staleWeights) > 0:
            for clientId in list(self.staleWeights):
                if (len(self.staleWeights[clientId]) > 0 and clientId in self.staleRemainDuration):
                    if self.staleRemainDuration[clientId] <= duration and self.stale_rounds[clientId] > 0:
                        if self.stale_rounds[clientId] <= args.stale_update or args.stale_update < 0:
                            round_stale_params[clientId] = [torch.from_numpy(param).to(device=self.device) for param in self.staleWeights[clientId]]
                            update_stale_rounds[clientId] = self.stale_rounds[clientId]
                            stale_count += 1

                            # Ahmed - account for stale resources that contributed to the model
                            self.stale_compute += self.stale_computation[clientId]
                            self.stale_communicate += self.stale_communication[clientId]
                            del self.stale_computation[clientId]
                            del self.stale_communication[clientId]

                            logging.info("==== Stale client {} apply update: remaining {} round duration {} stale rounds {} max stale {}".format(
                                    clientId, self.staleRemainDuration[clientId], duration, self.stale_rounds[clientId], args.stale_update))
                        else:
                            unused_stale += 1
                            logging.info("==== Stale client {} update expires:remaining {} round duration {} stale rounds {} max stale {}".format(clientId,\
                                                     self.staleRemainDuration[clientId], duration, self.stale_rounds[clientId], args.stale_update))
                        del self.staleWeights[clientId]
                        del self.staleRemainDuration[clientId]
                        del self.stale_rounds[clientId]
                        gc.collect()
                    else:
                        self.staleRemainDuration[clientId] -= duration
                        self.stale_rounds[clientId] += 1
                        logging.info("==== Stale client {} stay in cache: remaining {} duration {} stale rounds {} max stale {}".format(
                                clientId, self.staleRemainDuration[clientId], duration, self.stale_rounds[clientId], args.stale_update))
        logging.info("==== Stale Updates {} Count {} Unused {} Rounds {}".format(len(round_stale_params), stale_count, unused_stale, update_stale_rounds))
        return round_stale_params, update_stale_rounds, stale_count, unused_stale

    def stale_clients_handler(self):
        self.bad_update = self.good_update = self.bad_param = self.good_param = 0

        # Ahmed - update the remaining time or delete the stale update if updated already
        if len(self.round_stale_param) != self.round_stale_updates:
            logging.info(f'===== ROUND {self.epoch} AGREGATOR LOGICAL FATAL ERROR SKIPPING {len(self.round_stale_param)} vs {self.round_stale_updates}')

        elif self.round_update and self.round_stale_updates > 0:
            temp_stale_params = []
            stale_rounds = [x + 1 for x in self.update_stale_rounds]
            stale_rounds_sum = sum(stale_rounds) #sum(self.update_stale_rounds)
            stale_rounds_max = max(stale_rounds) #max(self.update_stale_rounds)

            self.update_clients.extend(list(self.round_stale_param.keys()))

            self.param_update_count += len(self.round_stale_param)

        del self.round_stale_param
        self.round_stale_param = {}


    def log_round_metrics(self):
        #logging.info(f'Aggregate label occurances: {self.label_counter}')
        # Ahmed - process values
        # Ahmed - log the train metrics information to file and wandb
        # rewards_dict = {x: clientSampler.getScore(0, x) for x in sorted(clientsLastEpoch)}
        # rewards_list = list(rewards_dict.values())
        #rem_durations = [x-self.round_duration for x  in list(self.staleRemainDuration.values())]
        rem_durations = [max(0, v) for k,v in list(self.staleRemainDuration.items()) if self.stale_rounds[k] > 0]
        stale_rounds_list = [v for v in list(self.stale_rounds.values()) if v > 0]

        #calculate model norm
        N1 = list()
        for idx, param in enumerate(self.model.parameters()):
            N1.append(np.linalg.norm(param.detach().numpy(), ord=None))
        agg_norm = sum(N1)

        # Ahmed - log scalars to wandb
        if args.use_wandb and self.epoch > 1:
            wandb.log({'Round/selected_clients': len(self.sampled_participants),
                       'Round/success_clients': self.tasks_round,
                       'Round/failed_clients': len(self.round_stragglers),
                       'Round/dropout_clients': len(self.dropout_clients),
                       'Round/priority_clients': len(self.priority_clients),
                       'Round/attended_clients': self.attended_clients,
                       'Round/unique_clients': len(self.unique_attend_clients),
                       'Round/online_clients': len(self.client_manager.cur_online_clients),
                       'Round/clock': self.global_virtual_clock,
                       'Round/epoch': self.epoch,
                       'Round/duration': self.round_duration,
                       'Round/mov_avg_deadline': self.mov_avg_deadline,
                       'Round/deadline': self.deadline if self.deadline > 0 else self.mov_avg_deadline,
                       'Round/compute': self.total_compute,
                       'Round/communicate': self.total_communicate,
                       'Round/new_compute': self.new_compute,
                       'Round/new_communicate': self.new_communicate,
                       'Round/stale_compute': self.stale_compute,
                       'Round/stale_communicate': self.stale_communicate,
                       'Round/param_norm': agg_norm,
                       #'Round/total_worker': self.total_worker,

                       #'Train/avg_loss': np.average(self.loss_accumulator),
                       'Train/acc_top_1': np.average(self.acc_accumulator) if len(self.acc_accumulator) > 0 else 0,
                       'Stats/std_top_1': np.std(self.acc_accumulator) if len(self.acc_accumulator) > 0 else 0,
                       'Stats/min_top_1': np.min(self.acc_accumulator) if len(self.acc_accumulator) > 0 else 0,
                       'Stats/max_top_1': np.max(self.acc_accumulator) if len(self.acc_accumulator) > 0 else 0,

                       'Train/acc_top_5': np.average(self.acc_5_accumulator) if len(self.acc_5_accumulator) > 0 else 0,
                       'Stats/std_top_5': np.std(self.acc_5_accumulator) if len(self.acc_5_accumulator) > 0 else 0,
                       'Stats/min_top_5': np.min(self.acc_5_accumulator) if len(self.acc_5_accumulator) > 0 else 0,
                       'Stats/max_top_5': np.max(self.acc_5_accumulator) if len(self.acc_5_accumulator) > 0 else 0,

                       'Train/loss': np.average(self.train_accumulator) if len(self.train_accumulator) > 0 else 0,
                       'Stats/std_loss': np.std(self.train_accumulator) if len(self.train_accumulator) > 0 else 0,
                       'Stats/min_loss': np.min(self.train_accumulator) if len(self.train_accumulator) > 0 else 0,
                       'Stats/max_loss': np.max(self.train_accumulator) if len(self.train_accumulator) > 0 else 0,

                       'Clients/avg_reward': np.average(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,
                       'Clients/std_reward': np.std(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,
                       'Clients/min_reward': np.min(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,
                       'Clients/max_reward': np.max(self.stats_util_accumulator) if len(
                           self.stats_util_accumulator) > 0 else 0,

                       'Clients/avg_completion': np.average(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,
                       'Clients/std_completion': np.std(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,
                       'Clients/min_completion': np.min(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,
                       'Clients/max_completion': np.max(self.completion_accumulator) if len(self.completion_accumulator) > 0 else 0,

                       'Stale/num_clients': len(self.staleWeights),
                       'Stale/avg_rem_duration': np.average(rem_durations) if len(rem_durations) else 0,
                       'Stale/std_rm_durations': np.std(rem_durations) if len(rem_durations) else 0,
                       'Stale/avg_rounds': np.average(stale_rounds_list) if len(stale_rounds_list) else 0,
                       'Stale/max_rounds': np.max(stale_rounds_list) if len(stale_rounds_list) else 0,
                       'Stale/min_rounds': np.min(stale_rounds_list) if len(stale_rounds_list) else 0,
                       'Stale/bad_update': self.bad_update,
                       'Stale/good_update': self.good_update,
                       'Stale/bad_param': self.bad_param,
                       'Stale/good_param': self.good_param,
                       'Stale/total_importance': float(sum(self.importance_list)),
                       'Stale/avg_importance': float(np.average(self.importance_list)),
                       }, step=self.epoch)

            #Ahmed - log the fairness of the selection process
            clients_selected = np.asarray(list(self.clients_select_count.values()))
            clients_success = np.asarray(list(self.clients_success_count.values()))
            clients_fail = np.asarray(list(self.clients_fail_count.values()))
            wandb.log({"Fairness/jain_selection": (1.0 / len(clients_selected) * (np.sum(clients_selected) ** 2) / np.sum(clients_selected ** 2)),
                       "Fairness/qoe_selection": (1.0 - (2.0 * clients_selected.std() / (clients_selected.max() - clients_selected.min()))),
                       "Fairness/jain_success": (1.0 / len(clients_success) * (np.sum(clients_success) ** 2) / np.sum(clients_success ** 2)),
                       "Fairness/qoe_success": (1.0 - (2.0 * clients_success.std() / (clients_success.max() - clients_success.min()))),
                       "Fairness/jain_fail": (1.0 / len(clients_fail) * (np.sum(clients_fail) ** 2) / np.sum(clients_fail ** 2)),
                        "Fairness/qoe_fail": (1.0 - (2.0 * clients_fail.std() / (clients_fail.max() - clients_fail.min()))),
                       }, step=self.epoch)

            #Ahmed - log stale updates
            # log updates from successful clients and stale updates applied in this round
            wandb.log({'Round/new_updates': self.tasks_round ,
                       'Round/stale_updates': self.round_stale_updates,
                       'Round/unused_stale': self.unused_stale,
                       'Round/round_failures': self.round_failures,
                       'Round/total_updates': self.total_updates},
                      step=self.epoch)

            # Ahmed - log histograms to wandb
            wandb.log({"Clients/train_acc": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(self.acc_accumulator), bins=10)),
                       "Clients/train_acc_5": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(self.acc_5_accumulator), bins=10)),
                       "Clients/train_loss": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(self.train_accumulator), bins=10)),
                       "Clients/rewards": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(self.stats_util_accumulator), bins=10)),
                       "Clients/completion_time": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(self.completion_accumulator), bins=10)),
                       "Clients/selection": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(clients_selected), bins=10)),
                       "Clients/success": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(clients_success), bins=10)),
                       "Clients/fail": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(clients_fail), bins=10)),
                       "Clients/importance": wandb.Histogram(np_histogram=np.histogram(np.nan_to_num(list(self.client_importance.values())), bins=10)),
                       }, step=self.epoch)

    def get_client_conf(self, clientId):
        # learning rate scheduler
        conf = {}
        conf['learning_rate'] = self.args.learning_rate
        return conf

    def normalize_importance(self):
        # Ahmed - Normalize the importance
        self.importance_list = list(self.client_importance.values())
        importance_sum = sum(self.importance_list)
        if self.args.stale_factor == 1 and importance_sum != self.total_updates:
            logging.info(f'======== FATAL ERROR round {self.epoch} importance sum {importance_sum} not equal {self.total_updates}')

        if self.round_update:
            for idx, param in enumerate(self.model.parameters()):
                if importance_sum > 0:
                    # Ahmed - update the model param by normalizing the importance
                    param.data /= 1.0 * importance_sum

        self.importance_list = [x / importance_sum for x in self.importance_list]
        importance_sum = sum(self.importance_list)
        if importance_sum > 1.01 or importance_sum < 0.99:
            logging.info(f'======== FATAL ERROR round {self.epoch} importance sum {importance_sum} not one')

    def update_importance(self):
        min_ratio = max_ratio = 1
        #Ahmed - update the stale client importance
        if len(self.round_stale_param) > 0 and self.args.stale_factor <= -4:
            self.client_ratio = {}
            for c in self.round_stale_param:
                val1 = val2 = 0
                for idx, param in enumerate(self.model.parameters()):
                    update = self.round_stale_param[c][idx]
                    val1 += torch.norm(update / (self.tasks_round + 1) + param.data / (
                                self.tasks_round + 1) - param.data / self.tasks_round) ** 2
                    val2 += torch.norm(param.data / self.tasks_round) ** 2
                ratio = abs(val1 / val2)
                self.client_ratio[c] = float(ratio)
            min_ratio = min(list(self.client_ratio.values()))
            max_ratio = max(list(self.client_ratio.values()))

        for c in self.round_stale_param:
            # Ahmed - just in case a value does not exist
            #divide by the stale factor
            if self.args.stale_factor > 1:
                self.client_importance[c] /= self.args.stale_factor

            #Equal - do nothing, the stale update gets the same weight as new one
            elif self.args.stale_factor == 1:
                pass

            # Average - divide by the average stale rounds
            elif self.args.stale_factor == -1:
                self.client_importance[c] /= np.average([self.update_stale_rounds[c] + 1 for c in self.update_stale_rounds])

            # AdaSGD
            elif self.args.stale_factor == -2:
                self.client_importance[c] /= (self.update_stale_rounds[c] + 1)

            # DynSGD
            elif self.args.stale_factor == -3:
                self.client_importance[c] *= math.exp(-(self.update_stale_rounds[c] + 1))

            #REFL method
            elif  self.args.stale_factor == -4:
                self.client_importance[c] *= (1 - self.args.stale_beta) / (self.update_stale_rounds[c] + 1) + \
                                             self.args.stale_beta * (1.0 - (math.exp(-self.client_ratio[c] / max_ratio) / self.scale_coff))

    def event_monitor(self):
        logging.info("AGGREGATOR - Start monitoring events ...")

        while True:
            if len(self.event_queue) != 0:
                event_msg = self.event_queue.popleft()
                send_msg = {'event': event_msg}

                if event_msg == 'update_model':
                    self.broadcast_msg(send_msg)
                    self.broadcast_models()

                elif event_msg == 'start_round':
                    for executorId in self.executors:
                        next_clientId = self.resource_manager.get_next_task()
                        if next_clientId is not None:
                            config = self.get_client_conf(next_clientId)
                            self.server_event_queue[executorId].put({'event': 'train', 'clientId':next_clientId, 'conf': config})

                elif event_msg == 'stop':
                    self.broadcast_msg(send_msg)
                    self.stop()
                    break

                elif event_msg == 'report_executor_info':
                    self.broadcast_msg(send_msg)

                elif event_msg == 'test':
                    self.broadcast_msg(send_msg)

                #Ahmed - handle the case when round failures is disabled and no client meets the deadline
                elif event_msg == 'skip_round':
                    if self.args.stale_skip_round:
                        # Ahmed - update the importance of the clients
                        self.update_importance()
                        logging.info(f'====== SKIP round complete - received new {self.tasks_round} strugglers {len(self.strugglers_to_run)}')

                        # Ahmed - invoke the stale clients handler first
                        self.stale_clients_handler()

                        # Ahmed - update the importance of the clients
                        self.normalize_importance()

                        logging.info(f'SKIP Round {self.epoch} updates {self.total_updates} importance list: {sum(self.importance_list)} {self.importance_list}')

                    self.round_completion_handler()

                    # Ahmed - log round metrics and reset
                    self.log_round_metrics()
                    self.round_reset_metrics()

            elif not self.client_event_queue.empty():

                event_dict = self.client_event_queue.get()
                event_msg, executorId, results = event_dict['event'], event_dict['executorId'], event_dict['return']

                if event_msg != 'train_nowait':
                    logging.info(f"Round {self.epoch}: Receive (Event:{event_msg.upper()}) from (Executor:{executorId})")

                # collect training returns from the executor
                if event_msg == 'train_nowait':
                    # pop a new client to run
                    next_clientId = self.resource_manager.get_next_task()

                    if next_clientId is not None:
                        config = self.get_client_conf(next_clientId)
                        runtime_profile = {'event': 'train', 'clientId':next_clientId, 'conf': config}
                        self.server_event_queue[executorId].put(runtime_profile)

                elif event_msg == 'train':

                    logging.info(f"Round {self.epoch}: Finished (Event:{event_msg.upper()}) Client {results['clientId']} from (Executor:{executorId})")

                    # push training results
                    self.client_completion_handler(results)#, importance=new_importance)

                    # Ahmed - perform handler if we have enough number of clients
                    total_clients = self.tasks_round +  len(self.strugglers_to_run)

                    if len(self.stats_util_accumulator) == total_clients:
                        logging.info(f'Round {self.epoch} updates {self.total_updates}:{self.param_update_count} client importance: {self.client_importance}')
                        # Ahmed - update the importance of the clients
                        self.update_importance()
                        logging.info(f'Round {self.epoch} updates {self.total_updates}:{self.param_update_count} client importance: {self.client_importance} client ratio: {self.client_ratio}')

                        #Ahmed - handle the collected deltas from clients
                        if self.args.send_delta and args.stale_update == 0:
                            self.client_deltas_handler()#importance=new_importance)

                        # Ahmed - invoke the stale clients handler first
                        self.stale_clients_handler() #importance=stale_importance)
                        logging.info(f'====== Aggregator update model - client IDs {len(self.update_clients)}{self.update_clients}')
                        logging.info(f'====== Aggregator round {self.epoch} complete - received {total_clients} new {self.tasks_round} strugglers {len(self.strugglers_to_run)} stale {len(self.round_stale_param)}:{self.total_updates} update count:{self.param_update_count}:{self.count_param_updates}')

                        #Ahmed - normalize importance and model param
                        self.normalize_importance()
                        logging.info(f'Round {self.epoch} updates {self.total_updates}:{self.param_update_count} importance list: {sum(self.importance_list)} {self.importance_list} param_update: {self.count_param_updates}')

                        self.round_completion_handler() #importance=stale_importance)

                        #Ahmed - log round metrics and reset
                        self.log_round_metrics()
                        self.round_reset_metrics()

                elif event_msg == 'test':
                    self.testing_completion_handler(results)

                elif event_msg == 'report_executor_info':
                    self.executor_info_handler(executorId, results)

                #Ahmed - stop the aggregator if one of the executors report trouble
                elif event_msg == 'executor_error':
                    logging.info("Aggregator received error message from executor".format(executorId))
                    self.broadcast_msg('emergency_stop')
                    logging.info("Aggregator broadcasted emergency stop due to error to executors")
                    time.sleep(5)
                    self.stop()
                    exit(-1)
                else:
                    logging.error("Unknown message types!")

            # execute every 100 ms
            time.sleep(0.1)

    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        #Ahmed - terminate after a minute and give wandb time to upload and shutdown wandb
        time.sleep(60)
        #Ahmed - stop the wandb run
        self.wandb_run.finish()

        self.control_manager.shutdown()

    def start_wandb(self):
        ############ Initiate WANDB ###############
        if args.use_wandb:
            # Ahmed - get wandb key if set
            if args.wandb_key is not None and str(args.wandb_key) != '' and str(args.wandb_key) != 'None':
                os.environ['WANDB_API_KEY'] = args.wandb_key
                logging.info('WANDB Key: {} {}'.format(os.environ['WANDB_API_KEY'], args.wandb_key))

            if args.wandb_key is not None and str(args.wandb_entity) != '' and str(args.wandb_entity) != 'None':
                os.environ['WANDB_ENTITY'] = args.wandb_entity
                logging.info('WANDB Entity: {} {}'.format(os.environ['WANDB_ENTITY'], args.wandb_entity))

            tags = []
            if args.wandb_tags is not None and str(args.wandb_tags) != '' and str(args.wandb_tags) != 'None':
                vals = str(args.wandb_tags).split('_')
                tags.extend(vals)


            # Ahmed - set and init wandb run
            project = args.data_set + '_' + args.model.split('_')[0] #+ '_' + len(args.learners.split('-'))
            gradient_policy = '_YoGi' if args.gradient_policy == 'yogi' else '_QFedAvg' if args.gradient_policy == 'qfedqvg' else '_Prox' if args.gradient_policy == 'prox' else '_FedAvg'
            run_name = 'Exp' + str(args.exp_type) + '_' + str(args.sample_mode) + str(gradient_policy) + '_S' + str(args.stale_update) + '_P' + str(args.avail_priority) + '_N' + str(args.total_worker) + '_D' + str(int(args.deadline)) + '_T' + str(
                args.target_ratio) + '_O' + str(args.overcommitment) + '_' + str(args.sample_seed) \
                # + '_R' + str(args.epochs) + '_E' + str(args.local_steps) + '_B' + str(args.batch_size)
            if args.resume_wandb:
                aggregator.wandb_run = wandb.init(settings=wandb.Settings(start_method="fork"), project=project, entity=args.wandb_entity, name=run_name,
                                                  config=dict(args.__dict__), id=run_name + '_' + args.time_stamp,
                                                  resume=True, anonymous='allow', tags=tags)
            else:
                aggregator.wandb_run = wandb.init(settings=wandb.Settings(start_method="fork"), project=project, entity=args.wandb_entity, name=run_name,
                                                  id=run_name + '_' + args.time_stamp, config=dict(args.__dict__),
                                                  anonymous='allow', tags=tags)
        ############################################

if __name__ == "__main__":

    aggregator = Aggregator(args)
    aggregator.run()

