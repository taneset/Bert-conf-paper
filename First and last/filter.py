#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from progress.bar import Bar
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

print('Loading Model...', end="")
model = AutoModelForSequenceClassification.from_pretrained("imdb_model", output_attentions=True, num_labels=2)
print('Done!')


def evaluation_loop(sample, quantile=0.05, attention_layer=-1):
    inputs = tokenizer.encode_plus(sample, return_tensors='pt', add_special_tokens=True, truncation=True)
    input_ids = inputs['input_ids']

    data = model(input_ids)
    initial_logits = data.logits
    attentions = data.attentions[attention_layer]

    attention_sums = attentions[0].sum(axis=0).sum(axis=0)
    threshold = attention_sums[1:attention_sums.shape[0] - 1].quantile(quantile)
    above_threshold = (attention_sums[1:attention_sums.shape[0] - 1] > threshold).nonzero(as_tuple=True)[0]
    above_threshold = torch.index_select(input_ids, 1, above_threshold + 1)
    above_threshold = torch.cat(
        [input_ids[:, 0].unsqueeze(1), above_threshold, input_ids[:, input_ids.shape[1] - 1].unsqueeze(1)], axis=1)

    data = model(above_threshold)
    final_logits = data.logits

    return initial_logits, final_logits, above_threshold


def evaluate_attention_filter(dataset, quantile, attention_layer=-1, save_text=False):
    print(f'Quantile {quantile}, Layer{attention_layer}')
    data_len = len(dataset)
    soft_m = torch.nn.Softmax(1)
    output = np.zeros((data_len, 2))

    with Bar('Processing...', max=data_len) as bar:
        for i in range(data_len):
            initial_text = dataset[i]['text']
            initial, final, above_threshold = evaluation_loop(initial_text, quantile=quantile,
                                                              attention_layer=attention_layer)
            output[i, 0] = soft_m(initial)[:, dataset[i]['label']]
            output[i, 1] = soft_m(final)[:, dataset[i]['label']]
            bar.next()

    return output


for quantile in [.05,.99,1.0]:
    output = evaluate_attention_filter(train_dataset, quantile)
    np.savetxt(f"train_last_th{quantile}.csv", output, delimiter=",")

    output = evaluate_attention_filter(eval_dataset, quantile)
    np.savetxt(f"eval_last_th{quantile}.csv", output, delimiter=",")

for quantile in [.05,.99,1.0]:
    output = evaluate_attention_filter(train_dataset, quantile, attention_layer=0)
    np.savetxt(f"train_first_th{quantile}.csv", output, delimiter=",")

    output = evaluate_attention_filter(eval_dataset, quantile, attention_layer=0)
    np.savetxt(f"eval_first_th{quantile}.csv", output, delimiter=",")

print('Complete')
