import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from model import GPTModel
import argparse
import parser_file
import torch
import torch.nn as nn
import tiktoken


def load_model(ags, path = None):
  model = GPTModel(ags.head_size, ags.num_layers, ags.num_embed, ags.time_step, ags.embed_size, ags.num_heads, ags.bias, ags.dropout).cuda()
  
  if path is not None:
    model_state_all = torch.load(path)
    model.load_state_dict(model_state_all["MODEL_STATE"])
  return model

def file_names(ags):
  
  
  file_name = "/data={}/model_type={}/batch_size={}/block_size={}/lr={}/save_every={}/dropout={}/bias={}/wd={}/beta1={}/\
    beta2={}/grad_clip={}/decay_lr={}/warmup_iters={}/lr_decay_iters={}/min_lr={}/".format(
              ags.data, ags.model_type, ags.batch_size, ags.time_step, ags.lr, ags.save_every, ags.dropout, ags.bias, ags.wd, ags.beta1, ags.beta2,\
                ags.grad_clip, ags.decay_lr, ags.warmup_iters, ags.lr_decay_iters, ags.min_lr)
      
  file_final = './ALLMODELS/AllModels_1/final'+file_name
  log_file = './ALLMODELS/AllModels_1/logfile'+file_name
  
  return file_final

if __name__ == "__main__":

  parser = argparse.ArgumentParser(add_help=False)

  parser_from_file = parser_file.initialize(parser)

  ags = parser_from_file.parse_args()
  
  
  if ags.model_type == 'gpt2': 
    ags.num_layers, ags.num_heads, ags.embed_size = 12, 12, 768 #135.68 M
  elif ags.model_type == 'gpt2-medium':
    ags.num_layers, ags.num_heads, ags.embed_size = 24, 16, 1024
  elif ags.model_type == 'gpt2-large':
    ags.num_layers, ags.num_heads, ags.embed_size = 36, 20, 1280
  elif ags.model_type == 'gpt2-xl':
    ags.num_layers, ags.num_heads, ags.embed_size = 48, 25, 1600
  else:
    raise('Please specify correct model type')
    
  ags.head_size = int(ags.embed_size / ags.num_heads)
  
  file_final = file_names(ags)
  
  test_model = load_model(ags, path = file_final + '/snapshot_val_loss.pt')
  
  test_model.eval()
  
  enc = tiktoken.get_encoding("gpt2")
  encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
  decode = lambda l: enc.decode(l)
  
  start = '\n'
  
  prompt = 'Lord Krishna is so merciful'
  
  start_idx = encode(prompt)
  x = (torch.tensor(start_idx, dtype=torch.long, device='cuda')[None, ...])
  print(x, len(x), x.shape[1])

  num_samples = 10
  max_new_tokens = 500
  temperature = 1.0
  top_k = 1
  
  #ags.time_step = x.shape[1]
  
  with torch.no_grad():
    text_file = open('./results.txt', 'w')

    for k in range(num_samples):
      
      y = test_model.generate(x, max_new_tokens, time_step = ags.time_step, temperature=temperature, top_k=top_k)
      y = y[y < 50257 ]
      if y.ndim > 1:
        line = decode(y[0].tolist()) 
      else:
        line = decode(y.tolist())         
      line = '{}:'.format(k+1) + line + '\n'
      text_file.write(line)
    
    text_file.close()
  