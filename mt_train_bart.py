import torch
from transformers import OPTForCausalLM, AutoTokenizer
from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacrebleu import corpus_bleu
import os
from dataclasses import dataclass, field
import json
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
import logging
import transformers
import copy
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import numpy as np 
import re


def collate_fn(batch):
    input_ids, labels,tgt_text,input_text,attention_mask= tuple([instance[key] for instance in batch] for key in ("input_ids", "labels",'tgt_text','input_text','attention_mask'))
    # print(input_ids)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # print(labels)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        tgt_text = tgt_text,
        input_text = input_text
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # print(strings)
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def find_checkpoint_folder(directory):
    pattern = r'^checkpoint_step_\d+$'
    
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)) and re.match(pattern, item):
            return directory+'/'+item
    
    return None

def preprocess_function(sources,targets,tokenizer):
    # inputs = [ex['de'] for ex in examples["translation"]]
    # targets = [ex['en'] for ex in examples["translation"]]
    model_inputs = tokenizer(sources, max_length=128,padding="longest", truncation=True,return_tensors="pt",)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128,padding="longest",truncation=True,return_tensors="pt",)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs['input_text'] = sources
    model_inputs['tgt_text'] = targets
    return model_inputs


class InstructionDataset_S(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,src='en',tgt='zh', split = 'train'):
        super(InstructionDataset_S, self).__init__()
        logging.warning("Loading data...")

        self.input_ids = []
        self.labels = []
        self.tgt_text = []
        self.input_text = []
        self.attention_mask = []

        data_path = '/home/ubuntu/ALMA/human_written_data/'+src+tgt+'/' +split+ '.'+src+'-'+tgt+'.json'
        src_data = []
        tgt_data = []
        data = {}
        if split == 'train':
            with open(data_path,'r') as f:
                l = f.readline()
                while(l):
                    d = json.loads(l)['translation']
                    src_data.append(d[src])
                    tgt_data.append(d[tgt])
                    l = f.readline()
        else:
            with open(data_path,'r') as f:
                l = f.readline()
                json_data = json.loads(l)
                for d in json_data:
                    src_data.append(d['translation'][src])
                    tgt_data.append(d['translation'][tgt])
          
            
        # instruction = 'Translate the sentence in {} to {}: '.format(mappings[src],mappings[tgt])

        # sources = [instruction+s +' ## Response: ' for s in src_data]
        sources = [s for s in src_data]
        targets = [f"{example}" for example in tgt_data]
        # print(len(targets))

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess_function(sources, targets, tokenizer)

        self.input_ids.extend(data_dict['input_ids'])
        self.labels.extend(data_dict['labels'])
        self.tgt_text.extend(data_dict['tgt_text'])
        self.input_text.extend(data_dict['input_text'])
        self.attention_mask.extend(data_dict['attention_mask'])
        print('The number of data samples: ' + str(len(self.input_ids)))
        print(len(self.input_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i],attention_mask = self.attention_mask[i],tgt_text = self.tgt_text[i],input_text = self.input_text[i])


import shutil

tlist = [['zhen','ruen','csen','deen'], \
    ['deen','csen','ruen','zhen'],
    ['csen','zhen','ruen','deen'],
    ['ruen','zhen','deen','csen'],
]

id = 0
tl = tlist[id]

output_path = 'seq_'+str(id)+'/'

torch.cuda.set_device(id)


mappings = {
        'en': 'en_XX',
        'zh': 'zh_CN',
        'de':'de_DE',
        'cs': 'cs_CZ',
        'ru': 'ru_RU'
    }




for i in range(len(tl)):
    task = tl[i]
    src = task[0:2]
    tgt = task[2:4]

    if os.path.exists(output_path+'/mbart_'+task):
        continue

    if i == 0 :
        model_name = 'facebook/mbart-large-50'
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=mappings[src], tgt_lang=mappings[tgt])

    else:
        model_name = find_checkpoint_folder(output_path+'/mbart_'+tl[i-1])
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=mappings[src], tgt_lang=mappings[tgt])

    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42) 

    IGNORE_INDEX = -100


    train_dataset = InstructionDataset_S(tokenizer=tokenizer,src=src,tgt = tgt,split = 'train')
    dev_dataset = InstructionDataset_S(tokenizer=tokenizer,src=src,tgt = tgt,split = 'valid')
    test_dataset = InstructionDataset_S(tokenizer=tokenizer,src=src,tgt = tgt,split = 'test')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=8,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16,collate_fn=collate_fn)

    from transformers import get_cosine_schedule_with_warmup
    # 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    from sacrebleu.metrics import BLEU

    def calculate_bleu(generated_texts, reference_texts):
        bleu = BLEU()
        return bleu.corpus_score(generated_texts, [reference_texts]).score


    # 翻译函数
    def translate(text):
        input_text = [f"{t}" for t in text]
        input_ids = tokenizer(input_text, return_tensors="pt",truncation=True,padding=True).input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=128, num_return_sequences=1, no_repeat_ngram_size=2)
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output


    # 评估函数
    def evaluate(model, dataloader):
        model.eval()
        predictions = []
        references = []
        bleu = BLEU()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                source_text = batch["input_text"]
                pred = translate(source_text)
                predictions.extend(pred)
                references.extend([[t] for t in batch["tgt_text"]])
        bleu_score = bleu.corpus_score(predictions, references).score

        return bleu_score


    # 训练循环
    num_epochs = 5
    best_bleu = -0.1
    best_checkpoint = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps = num_epochs*len(train_dataset)/8
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # # 保存最佳模型的函数
    def save_best_model(model, tokenizer, config,best_checkpoint,bleu):
        save_dir = best_checkpoint
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        model.save_pretrained(save_dir)
        
        # 保存tokenizer
        tokenizer.save_pretrained(save_dir)
        
        # 保存配置
        config.save_pretrained(save_dir)

        with open(save_dir+'/results.txt','a') as f:
            f.write(str(bleu)+'\n')
        
        print(f"Best model saved to {save_dir}")
        return save_dir

    import os

    print('~~~~~~~~~~~~Training ~~~~~~~~~~~~~')
    previous_model_dir = ''
    i = 1
    ls = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, 
                            labels=labels)
            loss = outputs.loss

            ls+=loss.item()

            loss.backward()
            optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
        
            if i%30 == 0:
                print('Loss function is :' + str(ls/30))
                ls = 0

            # 在开发集上评估
            if i%300 == 0:
                dev_bleu = evaluate(model, dev_loader)
                print(f"Epoch {epoch+1}, Dev BLEU: {dev_bleu}")
                
                if dev_bleu > best_bleu:
                    best_bleu = dev_bleu
                    best_checkpoint = f"{output_path}mbart_{src}{tgt}/checkpoint_step_{i}"
                    
                    if os.path.exists(previous_model_dir):
                        try:
                            shutil.rmtree(previous_model_dir)
                            print("Folder and all its contents deleted successfully")
                        except Exception as e:
                            print(f"Error: {e}")

                    best_model_dir = save_best_model(model, tokenizer, model.config,best_checkpoint,dev_bleu)
                    previous_model_dir = best_model_dir
            i+=1

    print(f"Best checkpoint: {best_checkpoint}, Dev BLEU: {best_bleu}")

    del model

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    model = MBartForConditionalGeneration.from_pretrained(best_checkpoint).cuda()
    test_bleu = evaluate(model, test_loader)
    print(f"Test BLEU: {test_bleu}")

    with open(best_checkpoint+'/results.txt','a') as f:
        f.write(str(test_bleu))
