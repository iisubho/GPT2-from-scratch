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


def ddp_setup():
  init_process_group(backend = "nccl")
  

class Trainer:
  def __init__(self,
               model:torch.nn.Module,
               train_data:None,
               val_data:None,
               loss_fn: torch.nn.CrossEntropyLoss,
               save_path:str,
               log_file:str,
               ags:argparse,
               )->None:
    
    self.local_rank = int(os.environ["LOCAL_RANK"])
    self.global_rank = int(os.environ["RANK"])
    self.model = model.to(self.local_rank)
    
    #print(ags.wd, 'wd', ags.lr, 'b', (ags.beta1,ags.beta2))
    #exit(1)
    
    self.optimizer = self.model.configure_optimizers(ags)

    self.train_data = train_data
    self.val_data = val_data
    self.loss_fn = loss_fn
    self.save_path = save_path
    self.iters_run = 0

    self.log_file = log_file
    self.ags = ags
    if os.path.exists(self.save_path+'/snapshot.pt'):

      self._load_snapshot(self.save_path+'/snapshot.pt')
      
    self.model = DDP(self.model, device_ids=[self.local_rank])    
  
  def _load_snapshot(self, snapshot_path):
    snapshot = torch.load(snapshot_path)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.iters_run = snapshot["Iter"]
    
    print(f"Resuming the model training from iterations {self.iters_run}")
    
  def _save_snapshot(self, Iter, name):
    snapshot = {}
    snapshot['MODEL_STATE'] = self.model.module.state_dict()
    snapshot['Iter'] = Iter
    snapshot['stats'] = self.stats
    snapshot['lr'] = self.updated_lr
    snapshot['optimizer'] = self.optimizer.state_dict()
    snapshot['ags'] = self.ags 
    snapshot['val_loss'] = self.val_loss1
    snapshot['train_loss'] = self.train_loss1
    
    torch.save(snapshot, name)
    
    
  def train(self, max_iters):
    train_loss, val_loss = torch.tensor([0.0]), torch.tensor([0.0])
    
    if not os.path.exists(self.log_file + '/results.txt'):      
      text_file = open(self.log_file + '/results.txt', 'w')
    else:
      text_file = open(self.log_file + '/results.txt', 'a')
      
    best_train_loss, best_val_loss = 1e6, 1e6
    
    self.stats = {
      'batch_loss': [], 'train_loss': [], 'val_loss': []
    }
        
    for e in tqdm(range(self.iters_run, max_iters)):
      self.model.train()

      self.updated_lr = self._get_lr(e)
      
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.updated_lr
        
      loss = self._run_iter(self.ags)        
      
      if self.local_rank == 0 and (e+1) == max_iters:
        self._save_checkpoint(name = self.save_path +'/final_iter={}.pt'.format(e+1))
          
      if (e+1) % self.ags.save_every == 0:
        val_loss = self._estimate_loss('val', self.ags)
        train_loss = self._estimate_loss('train', self.ags)
                
        if self.local_rank == 0:
        
          self.stats['batch_loss'].append(loss)
          self.stats['train_loss'].append(train_loss)
          self.stats['val_loss'].append(val_loss)
          
          self.val_loss1, self.train_loss1 = val_loss, train_loss 
          self._save_snapshot(e+1, name = self.save_path + '/snapshot.pt') 

          if train_loss < best_train_loss:
            best_train_loss = train_loss
            self._save_snapshot(e+1, name = self.save_path + '/snapshot_train_loss.pt')  
            
          if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            self._save_snapshot(e+1, name = self.save_path + '/snapshot_val_loss.pt')
          

      if self.local_rank == 0:
        #print(train_loss, val_loss, loss, 'ss')
        line = f'Iteration:{e+1} | Train loss : {round(train_loss.item(), 4)} | Val loss : {round(val_loss.item(), 4)} | batch loss : {round(loss, 4)}\n'
        text_file.write(line)

    text_file.close()


  def _save_checkpoint(self, name):    
    torch.save(self.model.module.state_dict(), name)
    
    
  def _run_iter(self, ags):
    self.optimizer.zero_grad(set_to_none = True)
    source, output = self._get_batch('train', ags) 
    source, output = source.to(self.local_rank), output.to(self.local_rank)
    _, loss = self.model(source, output, device=self.local_rank)
    loss.backward()
    if ags.grad_clip != 0.0:
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), ags.grad_clip)
      
    self.optimizer.step()
    return loss.item()

  def _get_lr(self, it):
    # 1) linear warmup for warmup_iters steps
    if it < self.ags.warmup_iters:
        return self.ags.lr * it / self.ags.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > self.ags.lr_decay_iters:
        return self.ags.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - self.ags.warmup_iters) / (self.ags.lr_decay_iters - self.ags.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return self.ags.min_lr + coeff * (self.ags.lr - self.ags.min_lr)
  
  @torch.no_grad()
  def _estimate_loss(self, split, ags):
    self.model.eval()
    losses = torch.zeros(ags.eval_iters)
    for k in range(ags.eval_iters):
      X, Y = self._get_batch(split, ags)
      X, Y = X.to(self.local_rank), Y.to(self.local_rank)
      _, loss = self.model(X, Y, device = self.local_rank)
      losses[k] = loss.item()
    out = losses.mean()
    self.model.train()
    return out




  def _get_batch(self, split, ags):
    data = self.train_data if split == 'train' else self.val_data
    ix = torch.randint(len(data) - ags.time_step, (ags.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+ags.time_step]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+ags.time_step]).astype(np.int64)) for i in ix])

    return x, y
  
  

def create_folder(ags):
  
  
  file_name = "/data={}/model_type={}/batch_size={}/block_size={}/max_iters={}/lr={}/save_every={}/dropout={}/bias={}/wd={}/beta1={}/\
    beta2={}/grad_clip={}/decay_lr={}/warmup_iters={}/lr_decay_iters={}/min_lr={}/".format(
              ags.data, ags.model_type, ags.batch_size, ags.time_step, ags.max_iter, ags.lr, ags.save_every, ags.dropout, ags.bias, ags.wd, ags.beta1, ags.beta2,\
                ags.grad_clip, ags.decay_lr, ags.warmup_iters, ags.lr_decay_iters, ags.min_lr)
      
  file_final = './ALLMODELS/AllModels_1/final'+file_name
  log_file = './ALLMODELS/AllModels_1/logfile'+file_name
  
  if not os.path.exists(file_final):
    
    try:
      os.makedirs(file_final)
    except FileExistsError:
      print('Already exist the main file\n')

  if not os.path.exists(log_file):
    
    try:
      os.makedirs(log_file)
    except FileExistsError:
      print('Already exist the log file\n')

  return file_final, log_file, file_name


  
  
def load_train_objs(ags):
  model = GPTModel(ags.head_size, ags.num_layers, ags.num_embed, ags.time_step, ags.embed_size, ags.num_heads, ags.bias, ags.dropout).cuda()
  return model
  
      
def loss_fnc():
  return nn.CrossEntropyLoss()

def main(ags, train_data, val_data, file_final, log_file):
  
  ddp_setup()
  
  model = load_train_objs(ags)

  loss_fn = loss_fnc()

    
  trainer = Trainer(model, train_data, val_data, loss_fn, save_path = file_final, log_file = log_file, ags = ags)

  trainer.train(ags.max_iter)   
    
  destroy_process_group()

    
if __name__ == "__main__":

  parser = argparse.ArgumentParser(add_help=False)

  parser_from_file = parser_file.initialize(parser)

  ags = parser_from_file.parse_args()
  
  file_final, log_file, file_name = create_folder(ags)
  
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
  

      
  train_data = np.memmap(os.path.join(ags.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
  val_data = np.memmap(os.path.join(ags.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
  
  main(ags, train_data, val_data, file_final, log_file)
  
  print('Finished training for {ags.max_iter} iterations')
