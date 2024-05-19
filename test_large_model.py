# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:18:59 2024

@author: Zidong
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

model_name = 'facebook/opt-125m'

torch_dtype = torch.float16

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
                    r'C:\research\MeZO\large_models\result\SST2-125m\checkpoint-2000',
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype
                )
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
SST2_dataset = load_dataset('glue', 'sst2')
test_sst2 = SST2_dataset['test']
# sample_data
sample_data = test_sst2[:2]
sentences, labels = sample_data['sentence'], sample_data['label']

tokens = tokenizer(sentences)

model(torch.tensor(tokens['input_ids']))