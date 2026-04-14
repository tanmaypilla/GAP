#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from model.baseline import TextCLIP
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from tools import *
from Text_Prompt import *
from KLLoss import KLLoss
import clip
import argparse

class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval('dict({})'.format(values))
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)

# --- HOCKEY TEXT PROMPT INTEGRATION ---
# Part-aware text embeddings: text_dict[aug_id] = tensor[num_classes, 77]
classes, num_text_aug, text_dict = text_prompt_hockey_pasta_pool_4part()
# Random synonym list for ind=0: pre-tokenized tensors
text_list = text_prompt_hockey_random()
print("Hockey text prompts loaded (part-aware PASTA + random synonyms).")
# ---------------------------------------

CLASS_NAMES = [
    "GLID_FORW", "ACCEL_FORW", "GLID_BACK", "ACCEL_BACK",
    "TRANS_FORW_TO_BACK", "TRANS_BACK_TO_FORW", "POST_WHISTLE_GLIDING",
    "FACEOFF_BODY_POSITION", "MAINTAIN_POSITION", "PRONE", "ON_A_KNEE",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.cuda.amp.GradScaler()

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/hockey/default.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeders.feeder_hockey.Feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8, 
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=64, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=1)
    parser.add_argument('--use-weighted-ce', type=str2bool, default=False)
    parser.add_argument('--use-balanced-sampler', type=str2bool, default=False)
    parser.add_argument(
        '--val-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for validation')

    return parser


class Processor():
    """ 
        Processor for Hockey Action Recognition (GAP)
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Safely handle missing 'debug' flag; default to False (i.e., normal training mode)
            debug_mode = arg.train_feeder_args.get('debug', False)
            if not debug_mode:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    # Automatically overwrite for convenience, or you can uncomment prompt
                    # answer = input('delete it? y/n:')
                    shutil.rmtree(arg.model_saved_name)
                    print('Dir removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                for name in self.arg.model_args['head']:
                    self.model_text_dict[name] = nn.DataParallel(
                        self.model_text_dict[name],
                        device_ids=self.arg.device,
                        output_device=self.output_device)
        
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            if self.arg.use_balanced_sampler:
                from feeders.balanced_sampler import BalancedBatchSampler
                balanced_sampler = BalancedBatchSampler(
                    train_dataset.label,
                    batch_size=self.arg.batch_size,
                    drop_last=True,
                )
                self.data_loader['train'] = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_sampler=balanced_sampler,
                    num_workers=self.arg.num_worker,
                    worker_init_fn=init_seed)
            else:
                self.data_loader['train'] = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker,
                    drop_last=True,
                    worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        if self.arg.val_feeder_args:
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)

        if self.arg.use_weighted_ce:
            weights_list = [
                0.20642959045719705, 0.26380245122370277, 1.2237966612053315,
                4.696176720475786, 3.9504002287021156, 4.152817430503381,
                4.878552515445719, 2.85831006308822, 4.324362384603348,
                33.05861244019139, 40.523460410557185
            ]
            class_weights = torch.FloatTensor(weights_list).cuda(output_device)
            self.loss_ce = nn.CrossEntropyLoss(weight=class_weights).cuda(output_device)
        else:
            self.loss_ce = nn.CrossEntropyLoss().cuda(output_device)

        self.loss = KLLoss().cuda(output_device)

        self.model_text_dict = nn.ModuleDict()

        # Load CLIP model (frozen)
        for name in self.arg.model_args['head']:
            print(f"Loading CLIP model: {name}")
            model_, preprocess = clip.load(name, device=self.output_device, jit=False)
            del model_.visual # We only need the text encoder
            model_text = TextCLIP(model_)
            model_text = model_text.cuda(self.output_device)
            self.model_text_dict[name] = model_text

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1]) # Fixed potential NameError here (arg -> self.arg)
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                [{'params': self.model.parameters(),'lr': self.arg.base_lr},
                {'params': self.model_text_dict.parameters(), 'lr': self.arg.base_lr*self.arg.te_lr_ratio}],
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        loss_ce_value = []
        part_names = ['global', 'head', 'hand', 'hip', 'foot']
        loss_te_per_part = {i: [] for i in range(5)}
        cosine_sim_per_part = {i: [] for i in range(5)}
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):            
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()

            # forward
            with torch.cuda.amp.autocast():
                # Get Skeleton Features
                output, feature_dict, logit_scale, part_feature_list = self.model(data)

                label = label.long().cuda(self.output_device)
                loss_ce = self.loss_ce(output, label)
                if self.arg.loss_alpha > 0:
                    label_g = gen_label(label.cpu())
                    loss_te_list = []

                    # Multi-modal Contrastive Loss Calculation
                    for ind in range(num_text_aug):
                        # Get Text Embeddings
                        if ind > 0:
                            # Use specific prompt type (e.g. Synonyms, PASTA)
                            text_id = np.ones(len(label), dtype=np.int8) * ind
                            texts = torch.stack([text_dict[j][i, :] for i, j in zip(label, text_id)])
                            texts = texts.cuda(self.output_device)
                        else:
                            # Use Random prompt
                            texts = list()
                            for i in range(len(label)):
                                text_len = len(text_list[label[i]])
                                text_id = np.random.randint(text_len, size=1)
                                text_item = text_list[label[i]][text_id.item()]
                                texts.append(text_item)
                            texts = torch.cat(texts).cuda(self.output_device)

                        text_embedding = self.model_text_dict[self.arg.model_args['head'][0]](texts).float()

                        if ind == 0:
                            # Global Contrastive Loss
                            logits_per_image, logits_per_text = create_logits(feature_dict[self.arg.model_args['head'][0]],text_embedding,logit_scale[:,0].mean())
                            ground_truth = torch.tensor(label_g,dtype=feature_dict[self.arg.model_args['head'][0]].dtype,device=device)
                        else:
                            # Part-Aware Contrastive Loss
                            logits_per_image, logits_per_text = create_logits(part_feature_list[ind-1],text_embedding,logit_scale[:,ind].mean())
                            ground_truth = torch.tensor(label_g,dtype=part_feature_list[ind-1].dtype,device=device)

                        loss_imgs = self.loss(logits_per_image,ground_truth)
                        loss_texts = self.loss(logits_per_text,ground_truth)
                        part_loss = (loss_imgs + loss_texts) / 2
                        loss_te_list.append(part_loss)

                        # Track per-part contrastive loss
                        loss_te_per_part[ind].append(part_loss.item())

                        # Track per-part cosine similarity
                        if ind == 0:
                            skel_feat = feature_dict[self.arg.model_args['head'][0]]
                        else:
                            skel_feat = part_feature_list[ind - 1]
                        with torch.no_grad():
                            sf_norm = skel_feat / skel_feat.norm(dim=-1, keepdim=True)
                            te_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                            cos_sim = (sf_norm * te_norm).sum(dim=-1).mean()
                        cosine_sim_per_part[ind].append(cos_sim.item())

                    loss = loss_ce + self.arg.loss_alpha * sum(loss_te_list) / len(loss_te_list)
                else:
                    loss = loss_ce

                loss_ce_value.append(loss_ce.item())

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

            # Per-step TensorBoard: CE and per-part contrastive losses
            if loss_ce_value:
                self.train_writer.add_scalar('loss_ce', loss_ce_value[-1], self.global_step)
            if self.arg.loss_alpha > 0:
                for pind in range(num_text_aug):
                    if loss_te_per_part[pind]:
                        self.train_writer.add_scalar(
                            f'loss_contrastive/{part_names[pind]}',
                            loss_te_per_part[pind][-1], self.global_step)
                        self.train_writer.add_scalar(
                            f'cosine_sim/{part_names[pind]}',
                            cosine_sim_per_part[pind][-1], self.global_step)

            timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))

        # Epoch-level per-part summary
        if loss_ce_value:
            mean_ce = np.mean(loss_ce_value)
            self.print_log('\tMean CE loss: {:.4f}'.format(mean_ce))
            self.train_writer.add_scalar('epoch_loss_ce', mean_ce, epoch)

        if self.arg.loss_alpha > 0:
            parts_summary = []
            for pind in range(num_text_aug):
                if loss_te_per_part[pind]:
                    mean_loss = np.mean(loss_te_per_part[pind])
                    mean_sim = np.mean(cosine_sim_per_part[pind])
                    parts_summary.append(f'{part_names[pind]}={mean_loss:.4f}(sim={mean_sim:.4f})')
                    self.train_writer.add_scalar(f'epoch_loss_contrastive/{part_names[pind]}', mean_loss, epoch)
                    self.train_writer.add_scalar(f'epoch_cosine_sim/{part_names[pind]}', mean_sim, epoch)
            self.print_log('\tPer-part contrastive: ' + ', '.join(parts_summary))

        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
            
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []  # <--- NEW: Collect labels here
            process = tqdm(self.data_loader[ln], ncols=40)
            
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label) # <--- NEW: Store labels
                
                with torch.no_grad():
                    b, _, _, _, _ = data.shape
                    data = data.float().to(self.output_device)
                    label = label.long().to(self.output_device)
                    
                    output, _, _, _ = self.model(data)
                    
                    loss = self.loss_ce(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    
                    _, predict_label = torch.max(output.data, 1)
                    step_log_interval = 5
            
            # Concatenate Results
            score = np.concatenate(score_frag)
            
            # <--- NEW ROBUST ACCURACY CALCULATION --->
            # Concatenate all labels collected from the loop
            total_labels = torch.cat(label_list).cpu().numpy()
            
            # Calculate Top-1 Accuracy manually
            # score shape: (N, C), total_labels shape: (N,)
            rank = score.argsort()
            hit_top1 = [l in rank[i, -1:] for i, l in enumerate(total_labels)]
            acc1 = sum(hit_top1) * 1.0 / len(hit_top1)
            
            # Calculate Top-3 Accuracy manually
            hit_top3 = [l in rank[i, -3:] for i, l in enumerate(total_labels)]
            acc3 = sum(hit_top3) * 1.0 / len(hit_top3)
            
            step = epoch + 1
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', np.mean(loss_value), step)
                self.val_writer.add_scalar('acc', acc1, step)

            # Track best validation accuracy
            if ln == 'val' and self.arg.phase == 'train':
                if acc1 > self.best_acc:
                    self.best_acc = acc1
                    self.best_acc_epoch = epoch + 1

            # Print Results
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            self.print_log('\tTop1: {:.2f}%'.format(100 * acc1))
            self.print_log('\tTop3: {:.2f}%'.format(100 * acc3))

            # Per-class and mean-class accuracy
            pred_labels = rank[:, -1]
            cm = confusion_matrix(total_labels, pred_labels, labels=range(self.arg.model_args['num_class']))
            per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
            mean_class_acc = per_class_acc.mean()

            self.print_log('\tMean class accuracy: {:.2f}%'.format(100 * mean_class_acc))
            for idx, name in enumerate(CLASS_NAMES):
                self.print_log('\t  {:>28s}: {:.2f}%'.format(name, 100 * per_class_acc[idx]))

            # Save per-class accuracy and confusion matrix to CSV
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(CLASS_NAMES)
                writer.writerow(per_class_acc)
                writer.writerows(cm)

            if save_score:
                if hasattr(self.data_loader[ln].dataset, 'sample_name'):
                    score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                    with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                        pickle.dump(score_dict, f)
                else:
                    self.print_log("\t[WARN] Skipping score saving: 'sample_name' not found in Feeder.")

        return acc1

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            eval_split = 'val' if 'val' in self.data_loader else 'test'

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=[eval_split])

            # If we never improved best_acc (e.g., no val split), fall back to last epoch
            if self.best_acc_epoch == 0:
                self.best_acc_epoch = self.arg.num_epoch

            self.print_log('Best {} accuracy: {:.2f}% at epoch {}'.format(
                eval_split, 100 * self.best_acc, self.best_acc_epoch))

            # test the best model
            pattern = os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*')
            ckpts = glob.glob(pattern)
            if not ckpts:
                self.print_log('No checkpoint found matching pattern: {}'.format(pattern))
                return
            weights_path = ckpts[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list and len(self.arg.device) > 1:
                weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.SafeLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()