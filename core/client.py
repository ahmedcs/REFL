import torch
import logging
import math
from utils.nlp import mask_tokens
from torch.autograd import Variable
#Ahmed - add quality evaluation helper modules
from utils.utils_model import accuracy
from utils.decoder import GreedyDecoder
import numpy as np
from fl_client_libs import *
import random
from collections import Counter


#Ahmed - for reporducbility - it does not work
#Ahmed - https://github.com/pytorch/pytorch/issues/7068
def setup_seed(self, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class Client(object):
    """Basic client component in Federated Learning"""
    def __init__(self, conf):
        pass

    def train(self, client_data, model, conf, round_num=0):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device


        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps * conf.batch_size)

        #Ahmed - add logic to make a copy of the model for sending model deltas
        if conf.gradient_policy == 'prox' or conf.send_delta :
            global_model = [param.data.clone() for param in model.parameters()]

        if conf.task == "detection":
            lr = conf.learning_rate
            params = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

        elif conf.task == 'nlp':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": conf.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=conf.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)

        #TODO: Ahmed - temporarily use vanilla SGD optimizer
        if args.enforce_sgd:
            optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate)

        if conf.task == 'voice':
             criterion = torch.nn.CTCLoss(reduction='none').to(device=device)
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0

        if conf.task == "detection":
            im_data = Variable(torch.FloatTensor(1).cuda())
            im_info = Variable(torch.FloatTensor(1).cuda())
            num_boxes = Variable(torch.LongTensor(1).cuda())
            gt_boxes = Variable(torch.FloatTensor(1).cuda())

        # Ahmed - set accuracy variables
        count=0
        train_loss = 0.0
        correct = 0.0
        top_5 = 0.0
        total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
        #Ahmed - Initiate the decoder for voice tasks
        decoder = None
        if conf.task == 'voice':
            decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

        #Ahmed - count the number of labels trained on
        label_counter = Counter()

        logging.info(f'Train client {clientId} on {len(client_data)} samples')

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            try:

                for data_pair in client_data:
                    if conf.task == 'nlp':
                        (data, _) = data_pair
                        data, target = mask_tokens(data, conf.tokenizer, conf, device=device)
                    elif conf.task == 'voice':
                        (data, target, input_percentages, target_sizes), _ = data_pair
                        input_sizes = input_percentages.mul_(int(data.size(3))).int()
                    elif conf.task == 'detection':
                        temp_data = data_pair
                        target = temp_data[4]
                        data = temp_data[0:4]
                    else:
                        (data, target) = data_pair

                    # Ahmed - update the counter of labels seen in training
                    label_counter.update(target.data.tolist())

                    if conf.task == "detection":
                        im_data.resize_(data[0].size()).copy_(data[0])
                        im_info.resize_(data[1].size()).copy_(data[1])
                        gt_boxes.resize_(data[2].size()).copy_(data[2])
                        num_boxes.resize_(data[3].size()).copy_(data[3])
                    elif conf.task == 'speech':
                        data = torch.unsqueeze(data, 1).to(device=device)
                    else:
                        data = Variable(data).to(device=device)

                    target = Variable(target).to(device=device)

                    if conf.task == 'nlp':
                        outputs = model(data, labels=target)
                        loss = outputs[0]

                        #ahmed - evalaute accuracy
                        train_loss += loss.data.item()
                        acc = accuracy(outputs[1].reshape(-1, outputs[1].shape[2]), target.reshape(-1), topk=(1, 5))
                        correct += acc[0].item()
                        top_5 += acc[1].item()

                    elif conf.task == 'voice':
                        outputs, output_sizes = model(data, input_sizes)
                        outputs = outputs.transpose(0, 1).float()  # TxNxH
                        loss = criterion(outputs, target, output_sizes, target_sizes)

                        # unflatten targets
                        split_targets = []
                        offset = 0
                        for size in target_sizes:
                            split_targets.append(target[offset:offset + size])
                            offset += size

                        #ahmed - evalaute accuracy
                        # Ahmed - obtain the word/char error rates
                        decoded_output, _ = decoder.decode(outputs, output_sizes)
                        target_strings = decoder.convert_to_strings(split_targets)

                        for x in range(len(target_strings)):
                            transcript, reference = decoded_output[x][0], target_strings[x][0]
                            wer_inst = decoder.wer(transcript, reference)
                            cer_inst = decoder.cer(transcript, reference)
                            total_wer += wer_inst
                            total_cer += cer_inst
                            num_tokens += len(reference.split())
                            num_chars += len(reference.replace(' ', ''))

                    elif conf.task == "detection":
                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        rois_label = model(im_data, im_info, gt_boxes, num_boxes)

                        loss = rpn_loss_cls + rpn_loss_box \
                                + RCNN_loss_cls + RCNN_loss_bbox

                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                    else:
                        output = model(data)
                        loss = criterion(output, target)

                        # Ahmed - add evaluation of accuracy
                        acc = accuracy(output, target, topk=(1, 5))
                        correct += acc[0].item()
                        top_5 += acc[1].item()

                    # ======== collect training feedback for other decision components [e.g., kuiper selector] ======
                    if conf.task == 'nlp':
                        loss_list = [loss.item()] #[loss.mean().data.item()]
                    elif conf.task == "detection":
                        loss_list = [loss.tolist()]
                        loss = loss.mean()
                    else:
                        loss_list = loss.tolist()
                        loss = loss.mean()

                    #Ahmed - sum up train loss
                    train_loss += np.average(loss_list)

                    temp_loss = sum([l**2 for l in loss_list])/float(len(loss_list))

                    # Ahmed - set the values for voice tasks
                    if conf.task == 'voice':
                        correct, top_5, test_len = float(total_wer), float(total_cer), float(num_tokens)

                    # only measure the loss of the first epoch
                    if completed_steps < len(client_data):
                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = temp_loss
                        else:
                            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * temp_loss

                    # ========= Define the backward loss ==============
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # ========= Weight handler ========================
                    if conf.gradient_policy == 'prox':
                        for idx, param in enumerate(model.parameters()):
                            param.data += conf.learning_rate * conf.proxy_mu * (param.data - global_model[idx])

                    completed_steps += 1
                    #Ahmed - count the target/samples
                    count += len(target)

                    if completed_steps == conf.local_steps:
                        break

            except Exception as ex:
                error_type = ex
                break

        logging.info(f'Finished Training of client {clientId} on {count} samples')

        #Ahmed - add logic to make a copy of the model for sending model deltas
        if conf.send_delta:
            model_param = [(param.data - global_model[idx].data).cpu().numpy() for idx, param in enumerate(model.parameters())]
        else:
            model_param = [param.data.cpu().numpy() for param in model.parameters()]

        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}

        #Ahmed - statistical utility of the client
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        # Ahmed - add train accuracy to results
        train_loss /= len(client_data)
        sum_loss = train_loss * count
        # Ahmed - calculate the train accuracy metrics
        if count > 0:
            acc = round(correct / float(count), 4)
            acc_5 = round(top_5 / float(count), 4)
            train_loss = round(train_loss, 4)
            results['train_acc'] = acc
            results['train_acc_5'] = acc_5
            results['train_loss'] = train_loss
            results['sample_count'] = count
        else:
            results['train_acc'] = 0
            results['train_acc_5'] = 0
            results['train_loss'] = 0
            results['sample_count'] = 0

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        # Ahmed - add the label counter to the returned train results
        results['label_counter'] = label_counter.items()

        #Ahmed - model weights
        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results


    def test(self, conf):
        pass


