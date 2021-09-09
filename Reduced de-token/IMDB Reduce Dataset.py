#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
from transformers import Trainer
import torch
from os import scandir, path, makedirs
import pandas as pd
from progress.bar import Bar


raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# In[16]:


train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

print('Loading Model...', end="")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", output_attentions=True, num_labels=2)
print('Done!')


# In[17]:


def get_subsample(input_ids, attention_sums, quantile):
    threshold = attention_sums[1:attention_sums.shape[0]-1].quantile(quantile)
    above_threshold = (attention_sums[1:attention_sums.shape[0]-1] > threshold).nonzero(as_tuple=True)[0]
    above_threshold = torch.index_select(input_ids, 1, above_threshold+1) 
    
    return above_threshold[0]
    
    
def gen_sample(sample, attention_layer=0):
    inputs = tokenizer.encode_plus(sample, return_tensors='pt', add_special_tokens=True, truncation=True)
    input_ids = inputs['input_ids']
    
    data = model(input_ids)
    initial_logits = data.logits
    attentions = data.attentions[attention_layer]

    attention_sums = attentions[0].sum(axis=0).sum(axis=0)
    top_50 = get_subsample(input_ids, attention_sums, .50)
    top_10 = get_subsample(input_ids, attention_sums, .90)
    top_1 = get_subsample(input_ids, attention_sums, .99)
    
    return tokenizer.batch_decode([top_50, top_10, top_1])


# In[21]:


for index, item in {'train':train_dataset, 'eval':eval_dataset}.items():
    current_list = []
    
    with Bar('Processing...', max=len(item)) as bar:
    
        for i in range(len(item)):
            sample = item[i]['text']
            processed = gen_sample(sample)
            processed.append(item[i]['label'])
            current_list.append(processed)
            bar.next()

    current_df = pd.DataFrame(current_list)
    current_df.to_csv(f'{index}.csv')


# In[ ]:




