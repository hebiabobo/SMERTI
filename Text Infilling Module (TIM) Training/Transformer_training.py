#!/usr/bin/env python
# coding: utf-8

# Based on code from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

# # Prelims

# In[ ]:


# from google.colab import drive
# from tqdm.autonotebook import get_ipython
#
# drive.mount('/content/drive')

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")
# get_ipython().run_line_magic('matplotlib', 'auto')

# In[ ]:


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if cuda:
    print("Using CUDA from GPU")

# get_ipython().system('nvidia-smi')


# # Model Architecture

# In[ ]:


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# In[ ]:


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ## Encoder and Decoder Stacks   

# In[ ]:


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[ ]:


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# In[ ]:


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# In[ ]:


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# In[ ]:


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# In[ ]:


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# In[ ]:


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# In[ ]:


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# ### Attention                                                                                                                                                                                                                                      

# In[ ]:


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# In[ ]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# ## Position-wise Feed-Forward Networks                                                                                                         

# In[ ]:


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ## Embeddings and Softmax                                                                                                                                                                                                                                               

# In[ ]:


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ## Positional Encoding                                                                                                                             
#                                                                                                                                                                                                                                                     
# 

# In[ ]:


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ## Full Model

# In[ ]:


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    model = model.cuda()
    return model


# # Training
# 

# ## Batches and Masking

# In[ ]:


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.cuda()
        return tgt_mask


# ## Training Loop

# In[ ]:


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch.src = batch.src.to(device)
        batch.trg = batch.trg.to(device)
        batch.src_mask = batch.src_mask.to(device)
        batch.trg_mask = batch.trg_mask.to(device)
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens).cuda()
        total_loss += loss
        total_loss = total_loss.cuda()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# ## Training Data and Batching

# In[ ]:


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    maximum = max(src_elements, tgt_elements)
    return maximum


# ## Optimizer                   

# In[ ]:


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# ## Regularization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#                                                                                                                                                                                                                                                                                                                       
# ### Label Smoothing
# 

# In[ ]:


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.criterion = self.criterion.to(device)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        target.data = target.data.to(device)
        true_dist = x.data.clone()
        true_dist = true_dist.to(device)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        mask = mask.to(device)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        self.true_dist = self.true_dist.to(device)
        final_criterion = self.criterion(x, Variable(true_dist, requires_grad=False)).to(device)
        return final_criterion


# ## Loss Computation

# In[ ]:


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x).cuda()
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss = loss.cuda()
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        final_loss = (loss.data * norm).cuda()
        return final_loss


# ## Greedy Decoding

# In[ ]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src_mask = src_mask.to(device)
    model = model.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# # Data Preparation

# In[ ]:


# For data loading.
import os
from torchtext import data, datasets

# from torchtext.data import Field, BucketIterator, TabularDataset

main_path = "/content/drive/My Drive/"
path24 = "D:/PycharmProjects/SMERTI-master/"

corpus_name = "News_Dataset"
corpus = os.path.join(main_path, corpus_name)
save_dir = os.path.join(corpus, "output/transformer")
data_path = os.path.join(corpus, "data/transformer_data")
os.chdir(data_path)

masked_train_path = os.path.join(data_path, "masked_train_headlines.txt")
unmasked_train_path = os.path.join(data_path, "train_headlines.txt")

masked_val_path = os.path.join(data_path, "masked_val_headlines.txt")
unmasked_val_path = os.path.join(data_path, "val_headlines.txt")

masked_test_path = os.path.join(data_path, "masked_test_headlines.txt")
unmasked_test_path = os.path.join(data_path, "test_headlines.txt")

# In[ ]:


from string import punctuation
import re

masked_train = open(masked_train_path, encoding='utf-8').read().split('\n')
unmasked_train = open(unmasked_train_path, encoding='utf-8').read().split('\n')

masked_val = open(masked_val_path, encoding='utf-8').read().split('\n')
unmasked_val = open(unmasked_val_path, encoding='utf-8').read().split('\n')

masked_test = open(masked_test_path, encoding='utf-8').read().split('\n')
unmasked_test = open(unmasked_test_path, encoding='utf-8').read().split('\n')


def process_text(s):
    s = s.lower().strip()
    s = re.sub('\!+', '!', s)
    s = re.sub('\,+', ',', s)
    s = re.sub('\?+', '?', s)
    s = re.sub('\.+', '.', s)
    s = re.sub("[^a-zA-Z.!?,\[\]'']+", ' ', s)
    for p in punctuation:
        if p not in ["'", "[", "]"]:
            s = s.replace(p, " " + p + " ")
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(pad_token=BLANK_WORD)
TGT = data.Field(init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

# In[ ]:


import pandas as pd

train_data = {'src': [line for line in masked_train], 'trg': [line for line in unmasked_train]}
train = pd.DataFrame(train_data, columns=["src", "trg"])

val_data = {'src': [line for line in masked_val], 'trg': [line for line in unmasked_val]}
val = pd.DataFrame(val_data, columns=["src", "trg"])

test_data = {'src': [line for line in masked_test], 'trg': [line for line in unmasked_test]}
test = pd.DataFrame(test_data, columns=["src", "trg"])

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

# In[ ]:


data_fields = [('src', SRC), ('trg', TGT)]
train, val, test = data.TabularDataset.splits(path=data_path, train='train.csv', validation='val.csv', test='test.csv',
                                              format='csv', fields=data_fields)

# In[ ]:


SRC.build_vocab(train, val)
TGT.build_vocab(train, val)


# ## Iterators

# In[ ]:


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    src, trg = src.cuda(), trg.cuda()
    return Batch(src, trg, pad_idx)


# ## Training

# In[ ]:


if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model = model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion = criterion.cuda()
    BATCH_SIZE = 4096
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

# ## Training the System

# In[ ]:


import time

start = time.time()

# model = torch.load('news_transformer_10_full.pt') #uncomment to continue training from a saved model (e.g. news_transformer_10_full.pt, set start=10)
start = 0  # change this depending on model loaded on previous line
log_f = open("SMERTI_loss_log.txt", 'a')
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
total_epochs = 15

for epoch in range(total_epochs):
    print("Beginning epoch ", epoch + 1 + start)
    model.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter),
              model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    with torch.no_grad():
        test_loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                              SimpleLossCompute(model.generator, criterion, None))
        print("validation_loss: ", test_loss)
    log_f.write(str(epoch + 1 + start) + ' | ' + str(test_loss) + '\n')

    path = "news_transformer_{}.pt".format((epoch + 1 + start))
    path2 = "news_transformer_{}_full.pt".format((epoch + 1 + start))
    torch.save(model.state_dict(), path)
    torch.save(model, path2)

log_f.close()

end = time.time()
print(end - start)

# <div id="disqus_thread"></div>
# <script>
#     /**
#      *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
#      *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
#      */
#     /*
#     var disqus_config = function () {
#         this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
#         this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
#     };
#     */
#     (function() {  // REQUIRED CONFIGURATION VARIABLE: EDIT THE SHORTNAME BELOW
#         var d = document, s = d.createElement('script');
#         
#         s.src = 'https://EXAMPLE.disqus.com/embed.js';  // IMPORTANT: Replace EXAMPLE with your forum shortname!
#         
#         s.setAttribute('data-timestamp', +new Date());
#         (d.head || d.body).appendChild(s);
#     })();
# </script>
# <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>
