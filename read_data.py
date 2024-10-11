import os
import pandas as pd
import numpy as np
import torch
from transformers import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import datasets
# from datasets import load_dataset, load_metric 
from pathlib import Path 

# task2 = ['yahoo','qqp','yelp','rte']

glue_task_list = {'rte':0,'cola':1,'mrpc':2,'qnli':3,'qqp':4,'mnli':5}
mbpa_task_list = {'ag':6,'yahoo':7,'yelp':8,'dbpedia':9}

def tokenize_function(examples, tokenizer, task,max_seq_len,offset, t_type='bert-base-uncased'):
    #  examples contains a batch of data.
    processed_examples = []
    if task in ["cola", "sst2"]:
        processed_examples = examples["sentence"]
    elif task in ["ax", "mnli", "mnli_mm"]:
        for p, h in zip(examples["sentence1"], examples["sentence2"]):
            processed_examples.append(p + str(tokenizer.sep_token) + h)
    elif task in ["mrpc", "rte", "stsb", "wnli"]:
        for s1, s2 in zip(examples["sentence1"], examples["sentence2"]):
            processed_examples.append(s1 + str(tokenizer.sep_token) + s2) 
    elif task in ["qnli"]:
        for q, s in zip(examples["question"], examples["sentence"]):
            processed_examples.append(q + str(tokenizer.sep_token) + s)    
    elif task in ["qqp"]:
        for s1, s2 in zip(examples["question2"], examples["question2"]):
            processed_examples.append(s1 + str(tokenizer.sep_token)+ s2)
    elif task in ["mnli_matched", "mnli_mismatched"]:
        raise ValueError("Use task=mnli instead. mnli_matched and mnli_mismatched only contains validation and test")
    else:
        raise ValueError("tokenization for task {} currently not supported".format(task)) 
    
    test_text = get_tokenized(processed_examples,tokenizer,max_seq_len,t_type=t_type)
    if task in ['rte','qnli']:
        labels = [int(u=='entailment')+offset for u in examples["label"]]
    elif task == 'mnli':
        labels_dic = {'neutral':0,'entailment':1,'contradiction':2}
        labels = [labels_dic[u]+offset for u in examples["label"]]
    else:
        labels = [int(u)+offset for u in examples["label"]]
    print(set(labels))
    dataset = myDataset(test_text,labels)
    return dataset


def process_mrpc(data_dir):
    examples = []
    with open(data_dir, encoding="utf-8-sig") as data_fh:
        columns = data_fh.readline().strip().split('\t')
        # print(columns)
        for row in data_fh:
            examples.append(row.strip().split('\t'))
    return pd.DataFrame(examples,columns=columns)


def senteval_load_file(filepath):
    """
    Input:
        filepath. e.g., "<repo_dir>/data/senteval/bigram_shift.txt"
    Return: 
        task_data: list of {'X': str, 'y': int}
        nclasses: int
    """
    # Just load all portions, and then do train/dev/test splitting myself
    filepath = filepath+'.txt'
    tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
    task_data=[]
    
    for linestr in Path(filepath).open().readlines():
        line = linestr.rstrip().split("\t")
        task_data.append({
            'X': line[-1], 'label': line[1]
        })

    # Convert labels str to int
    all_labels = [item['label'] for item in task_data]
    labels = sorted(np.unique(all_labels))
    tok2label = dict(zip(labels, range(len(labels))))
    nclasses = len(tok2label) 
    for i, item in enumerate(task_data):
        item['label'] = tok2label[item['label']]
    
    task_df ={'label': [u['label'] for u in task_data], 'text': [u['X'] for u in task_data]}

    task_df = pd.DataFrame(task_df)
    return task_df, nclasses


def constitute(data_df,tokenizer, n_train_per_class= 2000,
                       n_val_per_class=400, max_seq_len=256
                       , offset=0, t_type='bert'):
    train_idxs, val_idxs = train_val_split(data_df, n_train_per_class,
                                           n_val_per_class)
    
    train_labels, train_text = get_data_by_idx_con(data_df, train_idxs)
    val_labels, val_text = get_data_by_idx_con(data_df, val_idxs)
    train_labels = [label + offset for label in train_labels]
    val_labels = [label + offset for label in val_labels]
    train_text = get_tokenized(train_text, tokenizer, max_seq_len,t_type=t_type)
    val_text = get_tokenized(val_text, tokenizer, max_seq_len,t_type=t_type)
    train_dataset = myDataset(train_text, train_labels)
    val_dataset = myDataset(val_text, val_labels)

    print("#Train: {}".format(len(train_idxs)))
    print("#Test: {}".format(len(val_idxs)))
    return train_dataset, val_dataset

def load_glue(task_name,data_num=4000):
    data_dir =  '../data/glue_data/'+str(task_name) + '/'
    if task_name== "cola":
        train = pd.read_csv(data_dir+'train.tsv',sep='\t') 
        dev = pd.read_csv(data_dir+'dev.tsv',sep='\t') 
        names = ['a','label','b','sentence']
        train.columns = names
        dev.columns = names
    elif task_name == 'mrpc':
        train =process_mrpc(data_dir+'train.tsv') 
        dev = process_mrpc(data_dir+'dev.tsv') 
        train = train.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, errors="raise")
        dev = dev.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, errors="raise")
    elif task_name == 'mnli':
        train =process_mrpc(data_dir+'train.tsv') 
        dev = process_mrpc(data_dir+'dev.tsv') 
        train = train.rename(columns={"gold_label": "label"}, errors="raise")
        dev = dev.rename(columns={"gold_label": "label"}, errors="raise")
    else:
        train = pd.read_csv(data_dir+'train.tsv',sep='\t',header = 0,on_bad_lines = 'skip') 
        dev = pd.read_csv(data_dir+'dev.tsv',sep='\t',header = 0,on_bad_lines = 'skip')
        if task_name == 'qqp':
            train = train.rename(columns={"is_duplicate": "label"}, errors="raise")
            dev = dev.rename(columns={"is_duplicate": "label"}, errors="raise")
    label_id =  train.label
    train = train.drop('label',axis = 1)
    train.insert(0,'label',label_id)

    train_idx,_ = train_val_split(train, data_num,1)
    train = train.iloc[train_idx]    

    print("#Train: {}".format(len(train_idx)))
    print("#Test: {}".format(len(dev)))
    return train,dev

def prepare_datasets_models(task, tokenizer,batch_size,offset = 0, max_seq_len = 256, t_type='bert-base-uncased',data_num =4000):
    """
    Prepares everything needed for this experiment.
    Also reads from checkpoint if exists.
    """

    train,dev = load_glue(task,data_num) 

    train_dataset = tokenize_function(train,tokenizer, task, max_seq_len,offset,t_type)
    eval_dataset = tokenize_function(dev,tokenizer, task, max_seq_len,offset,t_type)

    return train_dataset,eval_dataset


def get_tokenized(texts, tokenizer, max_seq_len, t_type='bert-base-uncased'):
    result = []
    mask_res = []

    for text in texts:
        inputs = tokenizer(text, padding='max_length', max_length=max_seq_len,truncation=True)

        assert len(inputs['input_ids']) == max_seq_len
        assert len(inputs['attention_mask']) == max_seq_len

        result.append(inputs['input_ids'])
        mask_res.append(inputs['attention_mask'])
    return result, mask_res


def get_train_val_data(data_path,tokenizer, n_train_per_class= 2000,
                       n_val_per_class=1, max_seq_len=256
                       , offset=0, t_type='bert'):
    # print(model_class[2])
    # tokenizer = model_class[1].from_pretrained(model_class[2])
    # assert tokenizer.convert_tokens_to_ids('[PAD]') == 0, "token id error"
    # assert tokenizer.convert_tokens_to_ids('[CLS]') == 101, "token id error"
    # assert tokenizer.convert_tokens_to_ids('[MASK]') == 103, "token id error"

    data_path = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(data_path, header=None)

    train_idxs, val_idxs = train_val_split(train_df, n_train_per_class,
                                           n_val_per_class)
    train_labels, train_text = get_data_by_idx(train_df, train_idxs)
    val_labels, val_text = get_data_by_idx(train_df, val_idxs)

    train_labels = [label + offset for label in train_labels]
    val_labels = [label + offset for label in val_labels]

    train_text = get_tokenized(train_text, tokenizer, max_seq_len,t_type=t_type)
    val_text = get_tokenized(val_text, tokenizer, max_seq_len,t_type=t_type)

    train_dataset = myDataset(train_text, train_labels)
    val_dataset = myDataset(val_text, val_labels)

    print("#Train: {}".format(len(train_idxs)))
    return train_dataset, val_dataset


def get_test_data(data_path,tokenizer, max_seq_len=256,
                   offset=0, t_type='bert-base-uncased'):
    # tokenizer = model_class[1].from_pretrained(model_class[2])
    
    data_path = os.path.join(data_path, 'test.csv')
    test_df = pd.read_csv(data_path, header=None)
    test_idxs = list(range(test_df.shape[0]))
    np.random.shuffle(test_idxs)

    test_labels, test_text = get_data_by_idx(test_df, test_idxs)

    test_labels = [label + offset for label in test_labels]
    test_text = get_tokenized(test_text, tokenizer, max_seq_len,t_type=t_type)

    print("#Test: {}".format(len(test_labels)))
    test_dataset = myDataset(test_text, test_labels)
    return test_dataset


def train_val_split(train_df, n_train_per_class, n_val_per_class, seed=0):
    np.random.seed(seed)
    train_idxs = []
    val_idxs = []

    if 'label' not in train_df.columns:
        train_df = train_df.rename(columns={0: "label"}, errors="raise")
    # print(train_df)
    classes = set(train_df['label'])
    # min_class = min(train_df['label'])
    # max_class = max(train_df['label'])
    # print(min_class,max_class)
    for cls in classes:
        idxs = np.array(train_df[train_df['label'] == cls].index)
        # print(idxs)
        np.random.shuffle(idxs)
        train_pool = idxs[:-n_val_per_class]
        if n_train_per_class < 0:
            train_idxs.extend(train_pool)
        else:
            train_idxs.extend(train_pool[:n_train_per_class])
        val_idxs.extend(idxs[-n_val_per_class:])

    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)
    # print(train_idxs)
    return train_idxs, val_idxs


def get_data_by_idx(df, idxs):
    text = []
    labels = []
    for item_id in idxs:
        labels.append(df.loc[item_id, 0] - 1)
        text.append(df.loc[item_id, 2])
    return labels, text


def get_data_by_idx_con(df, idxs):
    text = []
    labels = []
    for item_id in idxs:
        labels.append(df.loc[item_id, 'label'])
        text.append(df.loc[item_id, 'text'])
    return labels, text


def prepare_dataloaders(data_dir, tasks, offsets, train_class_size,
                        val_class_size, train_batch_size, val_batch_size,
                        test_batch_size,model_class):
    task_num = len(tasks)
    train_loaders = []
    validation_loaders = []

    tokenizer = model_class[1].from_pretrained(model_class[2])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    for i in range(task_num):
        print('------Loading the dataset of ' + str(tasks[i])+'------')
        if tasks[i] == 'top_constituents':
            data_path = os.path.join(data_dir, tasks[i])
            data, label_num = senteval_load_file(data_path)
            # print(data
            train_dataset, test_dataset = \
                constitute(data,tokenizer, train_class_size,
                                400, offset=offsets[i],t_type=model_class[2])
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle=True, drop_last=True)
            validation_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                    shuffle=True, drop_last=True)
            train_loaders.append(train_loader)
            validation_loaders.append(validation_loader)

        if tasks[i] in mbpa_task_list:
            data_path = os.path.join(data_dir, tasks[i])
            train_dataset, val_dataset = \
                get_train_val_data(data_path,tokenizer, train_class_size,
                                1, offset=offsets[i],t_type=model_class[2])
            
            test_dataset = get_test_data(data_path,tokenizer, offset=offsets[i],t_type=model_class[2])
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle=True, drop_last=True)
            validation_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                    shuffle=True, drop_last=True)
            train_loaders.append(train_loader)
            validation_loaders.append(validation_loader)

        elif tasks[i] in glue_task_list:
            train_dataset, test_dataset =  prepare_datasets_models(tasks[i],tokenizer,train_batch_size,offsets[i],256,t_type=model_class[2],data_num=train_class_size)
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle=True)
            validation_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                    shuffle=True)
            train_loaders.append(train_loader)
            validation_loaders.append(validation_loader)
    return train_loaders, validation_loaders

def prepare_joint_dataloader(data_dir, tasks, offsets, train_class_size,
                        val_class_size, train_batch_size, val_batch_size,
                        test_batch_size,model_class):
    task_num = len(tasks)
    train_datasets = []
    train_loaders = []
    validation_loaders = []

    tokenizer = model_class[1].from_pretrained(model_class[2])

    for i in range(task_num):
        print('------Loading the dataset of ' + str(tasks[i])+'------')
        if tasks[i] in mbpa_task_list:
            data_path = os.path.join(data_dir, tasks[i])
            train_dataset, val_dataset = \
                get_train_val_data(data_path,tokenizer, train_class_size,
                                1, offset=offsets[i],t_type=model_class[2])
            
            test_dataset = get_test_data(data_path,tokenizer, offset=offsets[i],t_type=model_class[2])
            # train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
            #                         shuffle=True, drop_last=True)
            validation_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                    shuffle=True, drop_last=True)

            train_datasets.append(train_dataset)
            validation_loaders.append(validation_loader)
        elif tasks[i] in glue_task_list:
            train_dataset, test_dataset =  prepare_datasets_models(tasks[i],tokenizer,train_batch_size,offsets[i],256,t_type=model_class[2],data_num=train_class_size)
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle=True, drop_last=True)
            validation_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                    shuffle=True, drop_last=True)
            train_datasets.append(train_dataset)
            validation_loaders.append(validation_loader)

    train = merge_dataset(train_datasets)
    train_loader = DataLoader(train, batch_size=train_batch_size,
                                shuffle=True, drop_last=True)
    return train_loader, validation_loaders


class myDataset(Dataset):

    def __init__(self, data, labels):
        super(myDataset, self).__init__()
        self.text = data[0]
        self.mask = data[1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.text[idx], dtype=torch.long), \
               torch.tensor(self.mask[idx],dtype=torch.long),  torch.tensor(self.labels[idx],dtype=torch.long)
    
    def get_data_with_label(self,label):
        label_bl = [l==label for l in self.labels]
        txt = np.array(self.text)[np.array(label_bl)]
        lbs = np.array(self.mask)[np.array(label_bl)]
        return txt,lbs

        

def merge_dataset(d_list):
    d0 = d_list[0]
    for i in range(1,len(d_list)):
        d0.text.extend(d_list[i].text)
        d0.mask.extend(d_list[i].mask)
        d0.labels.extend(d_list[i].labels)
    return d0

def compute_class_offsets(tasks, task_classes):
    '''
    :param tasks: a list of the names of tasks, e.g. ["amazon", "yahoo"]
    :param task_classes:  the corresponding numbers of classes, e.g. [5, 10]
    :return: the class # offsets, e.g. [0, 5]
    Here we merge the labels of yelp and amazon, i.e. the class # offsets
    for ["amazon", "yahoo", "yelp"] will be [0, 5, 0]
    '''
    task_num = len(tasks)
    offsets = [0] * task_num
    prev = -1
    total_classes = 0
    for i in range(task_num):
        offsets[i] = total_classes
        total_classes += task_classes[i]
    return total_classes, offsets