import torch
import torch.nn as nn
from transformers import BertModel,RobertaModel, BertForPreTraining, GPT2ForSequenceClassification,GPT2Model,GPT2Config
import math
import torch.nn.functional as F

class BaseModel(nn.Module):

    def __init__(self, n_class, model_dir,hidden_output = False,model_type = BertModel):
        super().__init__()

        self.n_class = n_class
        self.net = model_type.from_pretrained(model_dir, output_hidden_states= hidden_output )
        print(model_type)
        self.hidden_output = hidden_output
        self.classifier = nn.Sequential(
            nn.Linear(1024, n_class)
        )

    def forward_with_hidden(self,x,mask):
        outputs = self.net(x,attention_mask=mask,output_hidden_states = False)
        hidden = outputs[0]
        w = hidden[:,0,:]
        logits = self.classifier(w)
        h =  F.normalize(w, p=2, dim=1)
        return logits,h

    def forward(self, x, mask):
        if self.hidden_output:   
            outputs = self.net(x,attention_mask=mask,output_hidden_states = True)
            return outputs
        outputs = self.net(x,attention_mask=mask,output_hidden_states = False)
        hidden = outputs[0]
        w = hidden[:,0,:]
        logits = self.classifier(w)
        return logits

class gpt2_Model(nn.Module):
    def __init__(self, n_class, model_dir,hidden_output = False,model_type = GPT2Model):
        super().__init__()
        self.n_class = n_class
        self.net = model_type.from_pretrained(model_dir, output_hidden_states= hidden_output)
        self.config = GPT2Config.from_pretrained(model_dir)
        self.hidden_output = hidden_output
        self.classifier = nn.Sequential(
            nn.Linear(self.config.n_embd, n_class)
        )

    def forward(self, x, mask):
        if self.hidden_output:   
            outputs = self.net(x,attention_mask=mask,output_hidden_states = True)
            return outputs
        if self.config.eos_token_id is None:
            self.config.eos_token_id = self.config.eos_token_id
        hidden_states = self.net(x,attention_mask=mask,output_hidden_states = False)[0]
        logits = self.classifier(hidden_states)
        batch_size, sequence_length = x.shape[:2]
        if x is not None:
            sequence_lengths = torch.eq(x, self.config.eos_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % x.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return pooled_logits