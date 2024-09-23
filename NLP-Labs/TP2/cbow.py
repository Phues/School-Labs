import torch
import torch.nn as nn
import torch.optim as optim
import spacy

class CBOW(nn.Module):
    def __init__(self, embedding_size, vocab_size=-1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).mean(1).squeeze(1) # batch_size * 4 * 100
        return self.linear(embeddings)
    
    