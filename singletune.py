import os
import argparse
from copy import deepcopy
from pathlib import Path 
from transformers import get_linear_schedule_with_warmup
import torch
from utils import EarlyStopping
from transformers import BertModel, \
    BertTokenizer,\
    RobertaModel,\
    RobertaTokenizer,\
    AlbertModel,\
    AlbertTokenizer,\
    GPT2ForSequenceClassification,\
    GPT2Tokenizer,\
    DistilBertModel,\
    DistilBertTokenizer,\
    XLNetModel,\
    XLNetTokenizer, \
    GPT2Model



glue_task_list = {'rte':0,'cola':1,'mrpc':2,'qnli':3,'qqp':4,'gold_label':5}
mbpa_task_list = {'ag':6,'yahoo':7,'yelp':8,'dbpedia':9}

MODEL_CLASSES = {
    'bert': (BertModel, BertTokenizer,'google-bert/bert-base-uncased'),
    "roberta": (RobertaModel,RobertaTokenizer,"FacebookAI/roberta-base"),
    "albert": (AlbertModel,AlbertTokenizer,"albert/albert-base-v2"),
    "distilbert": (DistilBertModel, DistilBertTokenizer,"distilbert/distilbert-base-uncased"),
    "gpt2": (GPT2Model,GPT2Tokenizer, "gpt2"),
}

task_lists = {
    1:['qqp', 'yahoo', 'cola', 'mnli','yelp','qnli'],
    2:[ 'qnli', 'yelp', 'mnli','cola','yahoo','qqp'],
    3:[ 'mnli', 'yahoo', 'qnli','qqp','yelp','cola'],
    4:['yahoo','yelp','mnli','cola'],
    5:['cola', 'mnli', 'yelp', 'yahoo'],
    6:['yelp', 'yahoo', 'cola', 'mnli'],
    7:[ 'cola', 'mnli', 'yahoo', 'qnli','qqp','yelp'],
    8:['top_constituents','mnli', 'yelp', 'yahoo'],
    9:['yahoo','yelp','mnli','top_constituents'],
}


DATA_DIR = '../data'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--epochs", nargs='+', type=int,
                    default=[10, 10, 10, 10, 10, 10, 10, 10],
                    help='Epoch number for each task')
parser.add_argument("--batch_size", type=int, default=32,
                    help='training batch size')
parser.add_argument("--model_class", type=str, default='albert',
                    help='model class for pretrained models')
parser.add_argument("--bert_learning_rate", type=float, default=3e-5,
                    help='learning rate for pretrained Bert')
parser.add_argument("--learning_rate", type=float, default=3e-5,
                    help='learning rate for Class Classifier')
parser.add_argument('--gpu', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n_labeled', type=int, default=2000,
                    help='Number of labeled data')
parser.add_argument('--tasks', nargs='+', type=str,
                    # default=['rte','cola','mrpc','yelp','qnli','ag'],
                    default=['ag','qnli','qqp','yelp','dbpedia','mnli','yahoo','cola'],
                    help='Task Sequence')
parser.add_argument('--probing',  type=str, default='yahoo',
                    help='Probing Task')
parser.add_argument('--probing_epoch', type=int, default=5)
parser.add_argument('--taskslist',  type=int,
                    default=7, help='Task Sequence')
parser.add_argument("--checkpoint_dir", type=str, default="_single_tasks_s=")

args = parser.parse_args()

torch.cuda.set_device(args.gpu)


args.tasks = task_lists[args.taskslist]

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from model import BaseModel,gpt2_Model
from read_data import compute_class_offsets, prepare_dataloaders

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
n_gpu = torch.cuda.device_count()
args.checkpoint_dir = '../results/'+ str(args.model_class)+'_'+str(len(args.tasks))+args.checkpoint_dir+str(args.seed)+'_t='+str(args.taskslist)+'/'



dataset_classes = {
    'yelp'  : 5,
    'yahoo'   : 10,
    'ag'      : 4,
    'rte':2,
    'cola':2,
    'mrpc':2,
    'qnli':2,
    'qqp': 2,
    'dbpedia':14,
    'mnli':3,
    'top_constituents':20
}


def train_step(model, optimizer, cls_CR, x,mask, y,lr_scheduler):
    model.train()
    logits = model(x,mask)
    loss = cls_CR(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    lr_scheduler.step()
    return loss


def validation(model, t, validation_loaders, test = False,loss_fct = None):
    '''
    Compute the validation accuracy on the first (t + 1) tasks,
    return the average accuracy over (t + 1) tasks and detailed accuracy
    on each task.
    '''
    model.eval()
    acc_list = []
    
    with torch.no_grad():
        avg_acc = 0.0
        total_loss = 0.0
        if test:
            for i in range(t + 1):
                valid_loader = validation_loaders[i]
                total = 0
                correct = 0
                for x, mask, y in valid_loader:
                    x,mask, y = x.to(device),mask.to(device), y.to(device)
                    batch_size = x.size(0)
                    logits = model(x,mask)
                    _, pred_cls = logits.max(1)
                    correct += pred_cls.eq(y.view_as(pred_cls)).sum().item()
                    total += batch_size
                print("acc on task {} : {} ".format(i, correct * 100.0 / total))
                avg_acc += correct * 100.0 / total
                acc_list.append(correct * 100.0 / total)
        else:
            valid_loader = validation_loaders[t]
            total = 0
            correct = 0
            for x, mask, y in valid_loader:
                x,mask, y = x.to(device),mask.to(device), y.to(device)
                batch_size = x.size(0)
                logits = model(x,mask)
                loss = loss_fct(logits, y)
                total_loss+=loss
                _, pred_cls = logits.max(1)
                correct += pred_cls.eq(y.view_as(pred_cls)).sum().item()
                total += batch_size
            print("acc on task {} : {}".format(t, correct * 100.0 / total))
            avg_acc += correct * 100.0 / total
            acc_list.append(correct * 100.0 / total)
            total_loss/=total
    return avg_acc, acc_list, total_loss

def main():
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    np.random.seed(0)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model_class = MODEL_CLASSES[args.model_class]

    task_num = len(args.tasks)
    task_classes = [dataset_classes[task] for task in args.tasks]
    total_classes, offsets = compute_class_offsets(args.tasks, task_classes)
    train_loaders, validation_loaders = \
        prepare_dataloaders(DATA_DIR, args.tasks, offsets, args.n_labeled,
                            args.n_labeled, args.batch_size, 32, 32,model_class)


    
    cls_CR = torch.nn.CrossEntropyLoss()

    best_model = None

    initial_model = gpt2_Model(total_classes,model_class[2],model_type = model_class[0]).to(args.device)


    

    for task_id in range(task_num):


        if task_id == 0:
            continue

        if os.path.exists(Path(args.checkpoint_dir,str(task_id)+str(args.tasks[task_id])+"_ckpt.pt")):
            print('Skip for tasks '+str(task_id))
            continue

        model = deepcopy(initial_model).to(args.device)

        early_stop = EarlyStopping(patience=20)
        stopping = False

        data_loader = train_loaders[task_id]
        length = len(data_loader)

        optimizer = AdamW(
            [
                {"params": model.net.parameters(), "lr": args.bert_learning_rate},
                {"params": model.classifier.parameters(), "lr": args.learning_rate},
            ]
        )

        lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs[task_id] * len(data_loader)
        )

        best_acc = 0
        best_model = deepcopy(model.state_dict())

        acc_track = []

        iteration = 1
        for epoch in range(args.epochs[task_id]):
            
            for x, mask, y in tqdm(data_loader, total=length, ncols=100):
                x,mask , y = x.to(device),mask.to(device), y.to(device)
                train_step(model, optimizer, cls_CR, x,mask, y, lr_scheduler)

                if iteration % 100 == 0:
                    print('\n')
                    print('Epoch '+str(epoch)+" : ----------------Validation-----------------")
                    avg_acc, acc_list,loss = validation(model, task_id, validation_loaders,loss_fct=cls_CR)
                    acc_track.append(acc_list)

                    if avg_acc > best_acc:
                        print("------------------Best Model Till Now------------------------")
                        best_acc = avg_acc
                        best_model = deepcopy(model.state_dict())
                    early_stop(avg_acc)
                    stopping = early_stop.early_stop
                
                    if stopping:
                        break
                iteration += 1

            if stopping:
                break

        model.load_state_dict(best_model)
        
        avg_acc, acc_list,_ = validation(model, task_id, validation_loaders,loss_fct=cls_CR)

        with open(Path(args.checkpoint_dir,str(task_id)+str(args.tasks[task_id])+'_r.txt'),'a') as f:
            f.write(str(acc_list)+'\n')
            f.close()

        if len(acc_track) > 0:
            print("ACC Track: {}".format(acc_track))
        
        torch.save(model.state_dict(),Path(args.checkpoint_dir,str(task_id)+str(args.tasks[task_id])+"_ckpt.pt"))






if __name__ == '__main__':
    print(args)
    main()
