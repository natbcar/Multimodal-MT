import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # return x's embedding vector（times math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))
        pe_pos = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)  # pe

    def forward(self, x):
        #  build pe w.r.t to the max_length
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


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


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # check the h number
        self.d_k = d_model // h
        self.h = h
        # 4 linear layers: WQ WK WV and final linear mapping WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # get batch size
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # parttion into h sections，switch 2,3 axis for computation.
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)  # final linear layer


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # rows
        std = x.std(-1, keepdim=True)
        x_zscore = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2*x_zscore+self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    SublayerConnection: connect Multi-Head Attention and Feed Forward Layers 
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.w_1(x)
        h2 = self.dropout(h1)
        return self.w_2(h2)


def clones(module, N):
    """
    "Produce N identical layers."
    Use deepcopy the weight are indenpendent.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers (blocks)"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, x_img=None):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask, x_img)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, mm_attn=None):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.mm_attn = mm_attn
        n_sublayers = 2
        if self.mm_attn is not None:
            print("using multimodal attention in encoder")
            n_sublayers = 3
        self.sublayer = clones(SublayerConnection(size, dropout), n_sublayers)
        self.size = size  # d_model

    def forward(self, x, mask, x_img=None):
        # X-embedding to Multi-head-Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        if self.mm_attn is not None:
            x = self.sublayer[1](x, lambda x: self.mm_attn(x, x_img, x_img))
        # X-embedding to feed-forwad nn
        return self.sublayer[-1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        "Generic N layer decoder with masking."
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, x_img=None):
        """
        Repeat decoder N times
        Decoderlayer get a input attention mask (src) 
        and a output attention mask (tgt) + subsequent mask 
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, x_img)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, mm_attn=None):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.mm_attn = mm_attn
        self.feed_forward = feed_forward
        n_sublayers = 3
        if self.mm_attn is not None:
            print("using multimodal attention in decoder")
            n_sublayers = 4
        self.sublayer = clones(SublayerConnection(size, dropout), n_sublayers)

    def forward(self, x, memory, src_mask, tgt_mask, x_img=None):
        m = memory # encoder output embedding
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        if self.mm_attn is not None:
            # print(x_img.shape)
            # print(x_img)
            x = self.sublayer[2](x, lambda x: self.mm_attn(x, x_img, x_img))
        # Context-Attention：q=decoder hidden，k,v from encoder hidden
        return self.sublayer[-1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, img_proj_enc, img_proj_dec):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator 
        self.img_proj_enc = img_proj_enc
        self.img_proj_dec = img_proj_dec

    def encode(self, src, src_mask, x_img=None):
        if x_img is not None and self.img_proj_enc is not None:
            x_img = self.img_proj_enc(x_img)
        return self.encoder(self.src_embed(src), src_mask, x_img)

    def decode(self, memory, src_mask, tgt, tgt_mask, x_img=None):
        if x_img is not None and self.img_proj_dec is not None:
            x_img = self.img_proj_dec(x_img)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, x_img)

    def forward(self, src, tgt, src_mask, tgt_mask, x_img=None):
        "Take in and process masked src and target sequences."
        # encoder output will be the decoder's memory for decoding
        return self.decode(self.encode(src, src_mask, x_img), src_mask, tgt, tgt_mask, x_img)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode: d_model to vocab mapping
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class View(nn.Module):
    """Helper class to be used inside Sequential object to reshape Variables"""
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class ImageLocalFeaturesProjector(nn.Module):
    """
        Reshape local image features.
    """
    def __init__(self, num_layers, nfeats, outdim, dropout=0.1,
            use_nonlinear_projection=False):
        """
        Args:
            num_layers (int): 1.
            nfeats (int): size of image features.
            outdim (int): size of the output dimension.
            dropout (float): dropout probablity.
            use_nonliner_projection (bool): use non-linear activation
                    when projecting the image features or not.
        """
        super(ImageLocalFeaturesProjector, self).__init__()
        assert(num_layers==1), \
                'num_layers must be equal to 1 !'
        self.num_layers = num_layers
        self.nfeats = nfeats
        self.dropout = dropout
        
        layers = []
        # reshape input
        layers.append( View(-1, 7*7, nfeats) )
        if use_nonlinear_projection:
            print('using non-linear projection head')
            # linear projection from feats to rnn size
            layers.append( nn.Linear(nfeats, 4098) )
            # if use_nonlinear_projection:
            layers.append( nn.ReLU() )
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(4098, outdim*num_layers))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        else:
            print("using linear projection head")
            layers.append(nn.Linear(nfeats, outdim*num_layers))
            layers.append(nn.Dropout(dropout))
        #self.batch_norm = nn.BatchNorm2d(512)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        return out


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')  # 2020 update
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


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
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))