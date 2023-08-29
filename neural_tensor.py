import torch
import torch.nn as nn

class  NeuralTensorNetwork(nn.Module):

    def __init__(self, dictionary_size, embedding_size, tensor_dim, dropout, device="cpu"):
        super(NeuralTensorNetwork, self).__init__()

        self.device = device
        self.emb = nn.Embedding(dictionary_size, embedding_size)
        self.tensor_dim = tensor_dim

        ##Tensor Weight
        # |T1| = (embedding_size, embedding_size, tensor_dim)
        self.T1 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T1.data.normal_(mean=0.0, std=0.02)

        # |T2| = (embedding_size, embedding_size, tensor_dim)
        self.T2 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T2.data.normal_(mean=0.0, std=0.02)

        # |T3| = (tensor_dim, tensor_dim, tensor_dim)
        self.T3 = nn.Parameter(torch.Tensor(tensor_dim * tensor_dim * tensor_dim))
        self.T3.data.normal_(mean=0.0, std=0.02)

        # |W1| = (embedding_size * 2, tensor_dim)
        self.W1 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W2| = (embedding_size * 2, tensor_dim)
        self.W2 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W3| = (tensor_dim * 2, tensor_dim)
        self.W3 = nn.Linear(tensor_dim * 2, tensor_dim)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, svo, sov_length):
        # |svo| = (batch_size, max_length)
        # |sov_length| = (batch_size, 3)

        svo = self.emb(svo)
        # |svo| = (batch_size, max_lenght, embedding_size)

        ## To merge word embeddings, Get mean value
        subj, verb, obj = [], [], []
        for batch_index, svo_batch in enumerate(sov_length):
            sub_svo = svo[batch_index]
            len_s, len_v, len_o = svo_batch
            subj += [torch.mean(sub_svo[:len_s], dim=0, keepdim=True)]
            verb += [torch.mean(sub_svo[len_s:len_s+len_v], dim=0, keepdim=True)]
            obj += [torch.mean(sub_svo[len_s+len_v:len_s+len_v+len_o], dim=0, keepdim=True)]

        subj = torch.cat(subj, dim=0)
        verb = torch.cat(verb, dim=0)
        obj = torch.cat(obj, dim=0)
        # |subj|, |verb|, |obj| = (batch_size, embedding_size)

        R1 = self.tensor_Linear(subj, verb, self.T1, self.W1)
        R1 = self.tanh(R1)
        R1 = self.dropout(R1)
        # |R1| = (batch_size, tensor_dim)

        R2 = self.tensor_Linear(verb, obj, self.T2, self.W2)
        R2 = self.tanh(R2)
        R2 = self.dropout(R2)
        # |R2| = (batch_size, tensor_dim)

        U = self.tensor_Linear(R1, R2, self.T3, self.W3)
        U = self.tanh(U)

        return U
