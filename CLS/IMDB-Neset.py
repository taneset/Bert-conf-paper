#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
from transformers import Trainer
from progress.bar import Bar
import torch

raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# In[2]:


train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

print('Loading Model...', end="")
model = AutoModelForSequenceClassification.from_pretrained("finetuned-bert", output_attentions=True, num_labels=2)
print('Done!')


# In[37]:


def get_means(sample):

    inputs = tokenizer.encode_plus(sample, return_tensors='pt', add_special_tokens=True, truncation=True)
    input_ids = inputs['input_ids']
    #print(input_ids)
    
    data = model(input_ids)
    attentions = data.attentions
    means = np.zeros(len(attentions))
    
    for i in range(len(attentions)):
        value = attentions[i].sum(0).sum(1)[:,0].mean()
        means[i] = value
    
    return means


# In[46]:


for index, item in {'train':train_dataset, 'eval':eval_dataset}.items():
    data = np.zeros([len(item), 12])
    
    with Bar('Processing...', max=len(item)) as bar:
    
        for i in range(len(item)):
            sample = item[i]['text']
            data[i,:] = get_means(sample)
            bar.next()
    
    np.savetxt(f"{index}.csv", data, delimiter=",")
    


# In[21]:





# In[30]:





# In[ ]:




