import torch
import torch.nn as nn

# Define the size of the vocabulary and the dimension of the embeddings
vocab_size = 10000  # Size of your vocabulary
embedding_dim = 300  # Dimension of the embedding vectors

# Create an embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Input tensor with integer indices
input_indices = torch.LongTensor([2, 5, 0, 7, 1])

# Get the embeddings for the input indices
embeddings = embedding_layer(input_indices)

print(embeddings)
print(embeddings.size())