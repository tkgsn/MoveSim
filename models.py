import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import torch
import numpy as np

def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])
    
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e10)
    
    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()
        
        self.linear = nn.Linear(out_dim, out_dim*3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        #in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y
        
        qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
        
        #break into n_heads
        q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        
        #n_heads => attention => merge the heads => mix information
        scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
        
        return out

class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x)))) 

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x


class SelfAttention(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, linear_dim, n_locations, n_embeddings, output_dim, seq_len, M1, M2, dropout=.1, add_residual=True):
        super().__init__()

        #model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.pe = PositionalEmbedding(embed_size, seq_len)
        self.n_embeddings = n_embeddings
        self.output_dim = output_dim
        self.n_locations = n_locations
        self.linear_dim = linear_dim
        
        #backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)
        
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, output_dim, bias=False)

        self.add_residual = add_residual

        if add_residual:
            self.linear_mat1_1 = nn.Linear(self.n_embeddings, self.linear_dim)
            self.linear_mat1_2 = nn.Linear(self.linear_dim, self.n_embeddings)

            self.linear_mat2_1 = nn.Linear(self.n_embeddings, self.linear_dim)
            self.linear_mat2_2 = nn.Linear(self.linear_dim, self.n_embeddings)
            
            self.M1 = M1
            self.M2 = M2

            # padding M1 and M2 with 0s so that their shapes are (n_embeddings, n_embedings)
            device = M1.device
            self.M1 = torch.cat((self.M1, torch.zeros((self.n_embeddings - self.M1.shape[0], self.n_locations)).to(device)), dim=0)
            self.M1 = torch.cat((self.M1, torch.zeros((self.n_embeddings, self.n_embeddings - self.M1.shape[1])).to(device)), dim=1)
            self.M2 = torch.cat((self.M2, torch.zeros((self.n_embeddings - self.M2.shape[0], self.n_locations)).to(device)), dim=0)
            self.M2 = torch.cat((self.M2, torch.zeros((self.n_embeddings, self.n_embeddings - self.M2.shape[1])).to(device)), dim=1)

            
    def forward_without_softmax(self, x):

        embed = self.embeddings(x)
        embed = embed + self.pe(embed)
        mask = torch.ones((embed.shape[1], embed.shape[1])).tril(diagonal=0).to(next(self.parameters()).device)
        # mask = None
        for encoder in self.encoders:
            embed = encoder(embed, mask=mask)
        embed = self.norm(embed)
        embed = self.linear(embed)

        if self.add_residual:
            
            mat1 = self.M1[x].type(torch.float32).reshape(-1, self.n_embeddings)
            mat1 = F.relu(self.linear_mat1_1(mat1))
            mat1 = torch.sigmoid(self.linear_mat1_2(mat1))
            mat1 = F.normalize(mat1).reshape_as(embed)
            
            mat2 = self.M2[x].type(torch.float32).reshape(-1, self.n_embeddings)
            mat2 = F.relu(self.linear_mat2_1(mat2))
            mat2 = torch.sigmoid(self.linear_mat2_2(mat2))
            mat2 = F.normalize(mat2).reshape_as(embed)

            # add residuals
            embed = embed + torch.mul(embed,mat1) + torch.mul(embed,mat2)

        return embed
            
    def forward(self, x, mask=None):
        # print("input", x)
        x = self.forward_without_softmax(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def forward_log_prob(self, x):
        x = self.forward_without_softmax(x)
        x = F.log_softmax(x, dim=-1)
        return x[:,-1,:]        
    

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len
    
    

class Discriminator(nn.Module):
    """Basic discriminator.
    """

    def __init__(self, traj_length, n_vocabs, embedding_dim=64, dropout=0.6,):
        super(Discriminator, self).__init__()
        num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160][:traj_length]
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10][:traj_length]
        self.embedding = nn.Embedding(num_embeddings=n_vocabs, embedding_dim=embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, embedding_dim)) for (n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sum(num_filters), 2)
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.embedding(x).unsqueeze(1)  
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                 for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + \
            (1. - torch.sigmoid(highway)) * pred
        pred = F.log_softmax(self.linear(self.dropout(pred)), dim=-1)
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)




def step(generator, i, output_dim, sample, end_index):

    input = sample.to(next(generator.parameters()).device)[:, :i+1].long()
    probs = torch.exp(generator.forward_log_prob(input)).detach().cpu().numpy()
    probs = probs[:,:output_dim] / probs[:,:output_dim].sum(axis=1, keepdims=True)

    for j, prob in enumerate(probs):
        if sample[j, i] == end_index:
            choiced = end_index
        else: 
            prob = prob / prob.sum()
            choiced = np.random.choice(output_dim, p=prob)
        sample[j, i+1] = choiced
    return sample


def make_frame_data(n_sample, seq_len, start_index):
    frame_data = torch.zeros((n_sample, seq_len+1))
    frame_data.fill_(start_index)
    return frame_data

def recurrent_step(generator, seq_len, output_dim, start_time, data, end_index):
    for i in range(start_time, seq_len):
        step(generator, i, output_dim, data, end_index)
    return data


def make_sample(batch_size, generator, n_sample, dataset, real_start=True, remove_end=True):

    frame_data = make_frame_data(n_sample, dataset.seq_len, dataset.IGNORE_IDX)
    start_time = 0
    
    if real_start:
        start_time = 0
        indice = np.random.choice(range(len(dataset)), n_sample, replace=False)
        # choose real data by indice from list
        real_data = [dataset.data[i] for i in indice]
        frame_data[:,0] = torch.tensor([v[0] for v in real_data])
        # frame_data[:,1] = torch.tensor(real_data[:,1])
    else:
        frame_data[:,0] = torch.tensor([dataset.START_IDX for _ in range(n_sample)])

    samples = []
    for i in range(int(n_sample / batch_size)):
        batch_input = frame_data[i*batch_size:(i+1)*batch_size]
        sample = recurrent_step(generator, dataset.seq_len, dataset.n_locations+1, start_time, batch_input, dataset.END_IDX).cpu().detach().long().numpy()
        samples.extend(sample)

    # remove end_idx
    generated_trajectories = []
    for sample in samples:
        if remove_end:
            generated_trajectories.append([v for v in sample if v != dataset.END_IDX])
        else:
            generated_trajectories.append([v for v in sample])

    if not real_start:
        generated_trajectories = [v[1:] for v in generated_trajectories]

    return generated_trajectories


def make_input_for_predict_next_location_on_all_stages(self, x, start_time=0):
    input = []
    for traj in x:
        for i in range(self.window_size):
            input.append([self.start_index]*(self.window_size-start_time-i) + [state.item() for state in traj[max(start_time-self.window_size+i,0):start_time+i]])

    return torch.tensor(input).long()

def predict_next_location_on_all_stages(self, x, start_time=0):
    input = self.make_input_for_predict_next_location_on_all_stages(x, start_time).to(next(self.parameters()).device)
    probs = []
    for i in range(int(len(input)/self.window_size)):
        windowed_input = input[i*self.window_size:(i+1)*self.window_size].to(next(self.parameters()).device)
        prob = self(windowed_input)
        probs.append(prob)
    return torch.cat(probs).reshape(x.shape[0]*self.window_size, -1)