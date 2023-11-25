import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class embedding(nn.Module):
  def __init__(self, num_embeddings, embed_size): #GPT-2 has vocab size of 50257, padded upto nearest multiple of 64.
      super(embedding, self).__init__()
      self.embed = nn.Embedding(num_embeddings, embed_size)
  
  def forward(self, x):
    return self.embed(x) #(B, T)--->(B, T, num_embeddings) @ (num_embeddings, embed_size) ---> (B, T, embed_size)
    
          
class MultiHeadSelfAttention(nn.Module):
  

  def __init__(self, head_size, embed_size, time_step, num_heads, dropout):
      super(MultiHeadSelfAttention, self).__init__()
      self.query = nn.Linear(embed_size, head_size)
      self.value = nn.Linear(embed_size, head_size)
      self.key = nn.Linear(embed_size, head_size)
      self.register_buffer('tril', torch.tril(torch.ones(time_step, time_step)))
      self.dropout = nn.Dropout(dropout)
      self.head_size = head_size
      self.num_heads = num_heads
      
      
  def forward(self, x):
    B, T, C = x.shape
    
    assert C == int(self.head_size * self.num_heads)
    x = x.unsqueeze(dim = 1).repeat(1, self.num_heads, 1, 1) #(B, n_h, T, C)
    
    q = self.query(x) #(B, n_h, T, C) @ (C, head_size) ---> (B, n_h, T, head_size)
    k = self.key(x) #(B, n_h, T, C) @ (C, head_size) ---> (B, n_h, T, head_size)
    v = self.value(x) #(B, n_h, T, C) @ (C, head_size) ---> (B, n_h, T, head_size)
    """C**-0.5 is multiplied in order to make the weight with 0 mean and 1 standard deviation.
    """ 
    weight = q @ k.transpose(-2, -1) * C**-0.5 #(B, T, n_h, head_size) @ (B, n_h, head_size, T) ----> (B, n_h, T, T)

    weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, n_h, T, T)
    weight = F.softmax(weight, dim = -1) #(B, n_h, T, T)
    weight = self.dropout(weight) #(B, n_h, T, T)

    avg = weight @ v #(B, n_h, T, T) @ (B, n_h, T, head_size) ---> (B, n_h, T, head_size)
    avg = avg.transpose(1, 2) # (B, T, n_h, head_size)
    avg = avg.contiguous().view(B, T, C) #(B, T, n_h*head_size), C = n_h*head_size
    avg = self.dropout(avg)
    return avg    
  
class FeedForward(nn.Module):
  def __init__(self, embed_size, bias, dropout):
      super(FeedForward, self).__init__()
      self.ln1 = nn.Linear(embed_size, 4 * embed_size, bias = bias)
      self.gelu = nn.GELU()
      self.ln2 = nn.Linear(embed_size * 4, embed_size, bias= bias)
      self.drop  = nn.Dropout(dropout)
      
  def forward(self, x):
    x = self.ln1(x)
    x = self.gelu(x)
    x = self.ln2(x)
    x = self.drop(x)
    
    return x
    
  
class Block(nn.Module):
  def __init__(self, head_size, embed_size, time_step, num_heads, dropout, bias):
      super(Block, self).__init__()
      self.layer_norm1 = nn.LayerNorm(embed_size, bias = bias)
      self.attention = MultiHeadSelfAttention(head_size, embed_size, time_step, num_heads, dropout)
      self.layer_norm2 = nn.LayerNorm(embed_size, bias = bias)
      self.feed_fwd = FeedForward(embed_size, bias, dropout)
      
  def forward(self, x):
    x = x + self.attention(self.layer_norm1(x)) #(B, T, C)
    x = x + self.feed_fwd(self.layer_norm2(x)) #(B, T, C)
    return x
    
  
class GPTModel(nn.Module):
  def __init__(self, head_size, num_layers, num_embeddings, time_step, embed_size, num_heads, bias, dropout):
      super(GPTModel, self).__init__()
      
      self.data_embed = embedding(num_embeddings, embed_size)
      self.pos_embed = embedding(time_step, embed_size)
      self.drop = nn.Dropout(dropout)
      self.Blocks = nn.Sequential(*[Block(head_size, embed_size, time_step, num_heads, dropout, bias) for _ in range(num_layers)])
      self.norm_layer = nn.LayerNorm(embed_size, bias = bias)
      
      self.vocab = nn.Linear(embed_size, num_embeddings, bias = False)
      
      for pn,p in self.named_parameters():
        if pn.endswith('c_proj.weight'):
          torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        
      print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

  def get_num_params(self, non_embedding=True):
      """Subtract the non embedding parameters from total parameters.
      """
      n_params = sum(p.numel() for p in self.parameters())
      if non_embedding:
          n_params -= sum(p.numel() for p in self.pos_embed.parameters())
      return n_params
      
  def forward(self, x, y = None, device=None):
    B, T = x.shape
    x_embed = self.data_embed(x) #(B, T, C)
    pos_embed = self.pos_embed(torch.arange(T).to(device)) ##(1, T, C)
    x = x_embed + pos_embed #(B, T, C)
    out = self.drop(x) #(B, T, C)
    out = self.Blocks(out) #(B, T, C)
    out = self.norm_layer(out) #(B, T, C)
    
    
    if y is not None:
      logits = self.vocab(out) #(B, T, vocab_size)
      loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index = -1) #scalar
    else: 
      logits = self.vocab(out) #(B, T, vocab_size)
      loss = None

    return logits, loss
  
  def configure_optimizers(self, ags):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': ags.wd},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=ags.lr, betas=(ags.beta1, ags.beta2), fused = True)

    return optimizer
  
  @torch.no_grad()
  def generate(self, idx, new_token_size, time_step, temperature = 1.0, top_k = None):
    
    for _ in range(new_token_size):
      idx_cond = idx if idx.shape[-1] < time_step else idx[:, -time_step:]
      
      logits, _ = self(idx_cond, device = 'cuda')
      
      logits = logits[:, -1, :]
      
      proba  =torch.softmax(logits, dim = -1)
      idx_next = torch.multinomial(proba, num_samples = 1)      
      idx = torch.cat((idx, idx_next), dim = -1)
    
    return idx

