# -*- coding: utf-8 -*-
import csv
import logging
import os
import pickle
import random
import time
from collections import Counter
# Ahmed - add new modules
from collections import OrderedDict
from pathlib import Path
from random import Random

import numpy as np
import torch
from argParser import args
from fllibs import *
from torch.utils.data import DataLoader

#set up the data generator to have consistent results
seed = 10
generator = torch.Generator()
generator.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = seed #torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets

        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass

        #Ahmed - set the number of samples per worker
        self.usedSamples = 0

        #Ahmed - introduce targets dict
        self.targets = OrderedDict()
        self.indexToLabel = {}

        # categarize the samples
        # last_label = None
        # count = 0
        for index, label in enumerate(self.labels):
            if label not in self.targets:
                self.targets[label] = []

            self.targets[label].append(index)
            self.indexToLabel[index] = label

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def trace_partition(self, data_map_file, ratio=1.0):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(len(self.data.data)):
            self.partitions[clientId_maps[idx]].append(idx)

        for i in range(len(unique_clientIds)):
            self.rng.shuffle(self.partitions[i])
            takelen = max(0, int(len(self.partitions[i]) * ratio))
            self.partitions[i] = self.partitions[i][:takelen]

    #Ahmed - add data mapping handlers (uniform, zipf, balanced) and class exclusion
    def partition_data_helper(self, num_clients, data_map_dir=None):
        tasktype = 'train' if not self.isTest else 'test'
        data_map_file = None
        if data_map_dir is not None:
            data_map_file = os.path.join(data_map_dir, tasktype + '.csv')
            #Ahmed - handle the case for reddit dataset where on IBEX mappings are stored on the metadata folder
            if args.data_set == 'reddit' or args.data_set == 'stackoverflow':
                data_map_dir = os.path.join(args.log_path, 'metadata', args.data_set, tasktype)
                data_map_file = os.path.join(data_map_dir,  'result_' + str(args.process_files_ratio) + '.csv')

        # Ahmed - apply ratio on the data - manipulate the data per uses
        ratio = 1.0
        if not self.isTest and self.args.train_ratio < 1.0:
            ratio = self.args.train_ratio
        elif self.isTest and self.args.test_ratio < 1.0:
            ratio = self.args.test_ratio

        # Ahmed - introduce the mapping based on other methods rather than read mapping file to partition trace
        if self.isTest:
            if self.args.partitioning < 0 or data_map_file is None or num_clients < args.total_worker:
                self.uniform_partition(num_clients=num_clients, ratio=ratio)
            else:
                self.trace_partition(data_map_file, ratio=ratio)
        elif self.args.partitioning <= 0:
            if self.args.partitioning < 0 or data_map_file is None:
                self.uniform_partition(num_clients=num_clients, ratio=ratio)
            else:
                self.trace_partition(data_map_file, ratio=ratio)
        else:
            self.custom_partition(num_clients=num_clients, ratio=ratio)

    def uniform_partition(self, num_clients, ratio=1.0):
        # random uniform partition
        numOfLabels = self.getNumOfLabels()
        #Ahmed - update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))
        logging.info(f"Uniform partitioning data, ratio: {ratio} applied for {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1. / num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def custom_partition(self, num_clients, ratio=1.0):
        # custom partition
        numOfLabels = self.getNumOfLabels()
        # Ahmed - update the data length to account for the ratio
        data_len = min(self.getDataLen(), int(self.getDataLen() * ratio))
        sizes = [1.0 / num_clients for _ in range(num_clients)]

        #get # of samples per worker
        #Ahmed - set the number of samples per worker
        self.usedSamples = self.args.used_samples if self.args.used_samples >= 0 else (self.args.batch_size + 1)
        # get number of samples per worker
        if self.usedSamples <= 0:
            self.usedSamples = int(data_len / num_clients)

        #Verify if the custom client partitioning exists
        num_class = numOfLabels
        num_remove_classes = 0
        if self.args.filter_class > 0:
            num_remove_classes = self.args.filter_class
        elif self.args.filter_class_ratio > 0:
            num_remove_classes = round(numOfLabels * (1 - self.args.filter_class_ratio))
        num_class -= num_remove_classes

        filename = 'part' + str(self.args.partitioning) + '_clients' + str(num_clients) + '_data' + str(data_len) + '_labels'\
                   + str(num_class) + '_samples' + str(self.usedSamples)

        folder = os.path.join(args.log_path, 'metadata', args.data_set, 'data_mappings')
        if not os.path.isdir(folder):
            Path(folder).mkdir(parents=True, exist_ok=True)

        custom_mapping_file = os.path.join(folder, filename)
        if args.this_rank != 1:
            while (not os.path.exists(custom_mapping_file)):
                time.sleep(120)
        if os.path.exists(custom_mapping_file):
            with open(custom_mapping_file, 'rb') as fin:
                logging.info(f'Loading partitioning from file {filename}')
                self.partitions = pickle.load(fin)
                for i, part in enumerate(self.partitions):
                    labels = [self.indexToLabel[index] for index in part]
                    #count_elems = Counter(labels)
                    #logging.info(f'part {i} len: {len(part)} labels: {count_elems.keys()} count: {count_elems.values()}')
            return

        #get targets
        targets = self.getTargets()
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}
        keyLength = [0] * numOfLabels
        for key in keyDir.keys():
            keyLength[keyDir[key]] = len(targets[key])

        logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels}:{num_class} labels on {num_clients} clients, use {self.usedSamples} sample per client ...")

        ratioOfClassWorker = self.create_mapping(sizes)

        if ratioOfClassWorker is None:
            return self.uniform_partition(num_clients=num_clients)

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])

        # split the classes
        for worker in range(len(sizes)):
            self.partitions.append([])
            # enumerate the ratio of classes it should take
            for c in list(targets.keys()):
                takeLength = int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]])
                takeLength = min(takeLength, keyLength[keyDir[c]])

                indexes = self.rng.sample(targets[c], takeLength)
                self.partitions[-1] += indexes
                labels = [self.indexToLabel[index] for index in self.partitions[-1]]
                count_elems = Counter(labels)
                tempClassPerWorker[worker][keyDir[c]] += takeLength

            logging.info(f'worker: {worker} created partition len: {len(self.partitions[-1])} class/worker: {sum(tempClassPerWorker[worker])} labels:{tempClassPerWorker[worker]} ratios: {ratioOfClassWorker[worker]}')
        del tempClassPerWorker

        #save the partitions as pickle file
        if not os.path.exists(custom_mapping_file):
            with open(custom_mapping_file, 'wb') as fout:
                 pickle.dump(self.partitions, fout)
            logging.info(f'Storing partitioning to file {filename}')

    def create_mapping(self, sizes):
        numOfLabels = self.getNumOfLabels()

        ratioOfClassWorker = None
        if self.args.partitioning == 1:
            ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)
        elif self.args.partitioning == 2:
            ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)
        elif self.args.partitioning == 3:
            ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)

        num_remove_class=0
        if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
            num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1 - self.args.filter_class_ratio))
            for w in range(len(sizes)):
                # randomly filter classes by forcing zero samples
                wrandom = self.rng.sample(range(numOfLabels), num_remove_class)
                for wr in wrandom:
                    ratioOfClassWorker[w][wr] = 0.0 #0.001

        logging.info("==== Class per worker partitioning:{} clients:{} labels:{} rem_lable:{} count:{}  ====\n {} \n".format(self.args.partitioning, len(sizes), numOfLabels, num_remove_class, np.count_nonzero(ratioOfClassWorker), repr(ratioOfClassWorker)))
        return ratioOfClassWorker

    def getTargets(self):
        tempTarget = self.targets.copy()
        #TODO:why the temp targets are reshuffled each time getTargets is called?
        for key in tempTarget:
             self.rng.shuffle(tempTarget[key])
        return tempTarget

    def log_selection(self,classPerWorker):
        totalLabels = [0 for i in range(len(classPerWorker[0]))]
        logging.info("====Total # of workers is :{}, w/ {} labels, {}".format(len(classPerWorker), len(classPerWorker[0]), len(self.partitions)))
        for index, row in enumerate(classPerWorker):
            rowStr = ''
            numSamples = 0
            for i, label in enumerate(classPerWorker[index]):
                rowStr += '\t'+str(int(label))
                totalLabels[i] += label
                numSamples += label
            logging.info(str(index) + ':\t' + rowStr + '\t' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index])))
            logging.info("=====================================\n")
        logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        logging.info("=====================================\n")


    def use(self, partition, istest):
        resultIndex = self.partitions[partition]

        exeuteLength = -1 if not istest else int(len(resultIndex) * self.args.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, isTest=False, collate_fn=None, seed=0):
    """Load data given client Id"""
    partition = partition.use(rank - 1, isTest)
    dropLast = False if isTest or (args.used_samples < 0) else True
    num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)