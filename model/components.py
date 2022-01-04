import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Embedding indices meanings:
# 0: padding for sequence per column
# 1: missing field for the sample
# 2/3/4/...: 1/2/3/... indices in the vocab file.

def init_weights(m):
    if 'Linear' in str(type(m)):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.xavier_normal_(m.weight, gain=1.)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.01)
    elif 'Embedding' in str(type(m)):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.xavier_normal_(m.weight, gain=1.0)
        print("embedding: " + str(m.weight.data))
        with torch.no_grad():
            m.weight[m.padding_idx].fill_(0.)
    elif 'ModuleDict' in str(type(m)):
        for param in module.values():
            nn.init.xavier_normal_(param.weight, gain=1.)
            with torch.no_grad():
                param.weight[param.padding_idx].fill_(0.)

class FieldEmbeddings(nn.Module):
    '''
    Set of embeddings for input fields
    '''
    def __init__(self, field_meta, vocab, emb_size, combiner_type = "none"):
        '''
        @input:
        - field_meta: {field_name: {field_type: ..., value_type: ..., field_enc: ...}}
        - vocab: {field_name: {values: idx}}
        - meb_size: embedding size of each field's value
        - combiner_type: value aggregator for each field
        '''
        super(FieldEmbeddings, self).__init__()
        self.field_enc_type = {f: desc["field_enc"] for f,desc in field_meta.items()}
        self.field_size = {f: len(vocab[f]) for f in field_meta if f in vocab}
        self.emb_size = emb_size
        self.combiner = combiner_type
        scalar_emb_dict, vector_emb_dict = {}, {}
        for f,V in self.field_size.items():
            if self.field_enc_type[f] == "v2id" or self.field_enc_type[f] == "v2multid":
                scalar_emb_dict[f] = nn.Embedding(V+1, 1, padding_idx=0, sparse=False)
                vector_emb_dict[f] = nn.Embedding(V+1, self.emb_size, padding_idx=0, sparse=False)
            elif self.field_enc_type[f] == "v2onehot":
                scalar_emb_dict[f] = nn.Linear(V+1, 1)
                vector_emb_dict[f] = nn.Linear(V+1, self.emb_size)
            else:
                raise NotImplemented
            init_embedding(scalar_emb_dict[f])
            init_embedding(vector_emb_dict[f])
        self.scalar_embeddings = nn.ModuleDict(scalar_emb_dict)
        self.vector_embeddings = nn.ModuleDict(vector_emb_dict)
        
    def get_emb(self, field_name, values, N):
        # [bsz, N, L]
        scalar_embs = self.scalar_embeddings[field_name](values).view(values.shape[0],N,-1)
        # [bsz, N, L, d]
        vector_embs = self.vector_embeddings[field_name](values).view(values.shape[0],N,-1,self.emb_size)
        if self.combiner == "sum":
            # [bsz, N, 1]
            scalar_embs_combined = torch.sum(scalar_embs, dim = 2, keepdim = True)
            # [bsz, N, 1, d]
            vector_embs_combined = torch.sum(vector_embs, dim = 2, keepdim = True)
        else:
            scalar_embs_combined = scalar_embs
            vector_embs_combined = field_embs
        return scalar_embs_combined, vector_embs_combined

    def forward(self, features):
        """
        @input:
        - features: emb vocab ids, {field_name: [bsz,N,L]}
        @output:
        - return_embs: {"scalar": {field_name: [bsz,N,K]}, "vector": {field_name: [bsz,N,K,d]}}
                where K = 1 if combiner == "sum", K = L otherwise
        """
        return_embs = {"scalar": {}, "vector": {}}
        N = features["N"]
        # scalar embeddings
        for f in self.field_enc_type:
            scalar, vector = self.get_emb(f, features[f], N)
            return_embs["scalar"][f] = scalar
            return_embs["vector"][f] = vector
        return return_embs
    
    @staticmethod
    def combine_fields(emb_dict, combiner = "concat"):
        '''
        @input:
        - emb_dict: {field_name: (B,N,L)}}
        - combiner: one of ["concat", "average"]
        @output:
        - field_scalar: (B,N,K) if combiner == "concate", (B,N) if combiner == "average"
        - field_vector: (B,N,K,d) if combiner == "concate", (B,N,d) if combiner == "average"
        '''
        if combiner == "concat":
            # (B,N,K,d)
            combined_emb = torch.cat([v for v in emb_dict.values()], dim = 2)
        elif combiner == "average":
            # (B,N,d)
            combined_emb = torch.mean(torch.stack([torch.mean(v, dim = 2) for v in emb_dict.values()], dim = 2), dim = 2)
        return combined_emb
            
class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim = 1, dropout_rate = 0., do_batch_norm = True):
        super(DNN, self).__init__()
        self.in_dim = in_dim
        layers = []

        # hidden layers
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(in_dim, hidden_dim)
            # torch.nn.init.xavier_uniform_(linear_layer.weight, gain=nn.init.calculate_gain('relu'))
            layers.append(linear_layer)
            in_dim = hidden_dim

            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            if do_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

        # prediction layer
        last_layer = nn.Linear(in_dim, out_dim)
        layers.append(last_layer)
        # torch.nn.init.xavier_uniform_(last_layer.weight, gain=1.0)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        @input:
            `inputs`, [bsz, in_dim]
        @output:
            `logit`, [bsz, out_dim]
        """
        inputs = inputs.view(-1, self.in_dim)
        logit = self.layers(inputs)
        return logit
    
class AsymmetricMLP(nn.Module):
    def __init__(self, in_dim_left, in_dim_right, hidden_dims, out_dim = 1, dropout_rate = 0., do_batch_norm = True):
        super(AsymmetricMLP, self).__init__()
        self.in_dim_left = in_dim_left
        self.in_dim_right = in_dim_right
        if len(hidden_dims) > 0:
            self.first_layer_left = nn.Linear(in_dim_left, hidden_dims[0])
            self.first_layer_right = nn.Linear(in_dim_right, hidden_dims[0])
        else:
            self.first_layer_left = nn.Linear(in_dim_left, out_dim)
            self.first_layer_right = nn.Linear(in_dim_right, out_dim)

        layers = []
        in_dim = hidden_dims[0]
        # hidden layers
        for hidden_dim in hidden_dims[1:] + [out_dim]:
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            if do_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            linear_layer = nn.Linear(in_dim, hidden_dim)
            # torch.nn.init.xavier_uniform_(linear_layer.weight, gain=nn.init.calculate_gain('relu'))
            layers.append(linear_layer)
            in_dim = hidden_dim
        # torch.nn.init.xavier_uniform_(last_layer.weight, gain=1.0)

        self.remaining_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        @input:
            `inputs`, {'left': [..., in_dim_left], 'right': [..., in_dim_right]}
        @output:
            `logit`, [..., out_dim]
        """
        user_hidden = self.first_layer_left(inputs["left"]) + self.first_layer_right(inputs["right"])
        logit = self.remaining_layers(user_hidden)
        return logit
    
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class TargetAwareAttention(nn.Module):
    '''
    Target-aware attention
    '''
    def __init__(self, p):
        self.p = p
        super(TargetAwareAttention, self).__init__()
    
    def forward(self, inputs):
        # (B,N,1,d), (B,1,K,d)
        target_repr, multi_repr = inputs['target_repr'], inputs['multi_repr']
        # (B,N,K)
        kq_weights = torch.sum(target_repr * multi_repr, dim = -1)
        attn = F.softmax(torch.pow(kq_weights, self.p), dim = -1)
        # (B,N,d)
        combined_repr = torch.sum(multi_repr * attn[:,:,:,None], dim = -2)
        # (B,N)
        scores = torch.sum(combined_repr.view(target_repr.shape) * target_repr, dim = -1)
        return scores, combined_repr, attn

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

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, key_dim, value_dim, x_dim, dropout=0.1):
        '''
        Take in model size and number of heads.
        '''
        super().__init__()
#         assert d_model % h == 0

        # We assume d_v always equals d_k
#         self.d_k = d_model // h
        self.d_k = key_dim
        self.d_v = value_dim
        self.d_x = x_dim
        self.h = h

        self.key_linear_layer = nn.Linear(self.d_x, self.d_k * self.h)
        self.query_linear_layer = nn.Linear(self.d_x, self.d_k * self.h)
        self.value_linear_layer = nn.Linear(self.d_x, self.d_v * self.h)
        self.output_linear = nn.Linear(self.d_v * self.h, self.d_x)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        @input:
        - query/key/value: [bsz, -1, x_dim]
        '''
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.query_linear_layer(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.key_linear_layer(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.value_linear_layer(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v * self.h)
        return self.output_linear(x)