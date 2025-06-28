import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

  def __init__(self, d_model: int, vocab_size: int):
    super().__init__() # Call the parent constructor
    self.embeddings = d_model # Size of the embeddings
    self.vocab_size = vocab_size # Size of the vocabulary
    self.embeddings = nn.Embedding(vocab_size, d_model) # Create the embedding layer

  def forward(self, x): # Forward pass through the embedding layer
    return self.embeddings(x)  * math.sqrt(self.d_model) # Scale the embeddings by the square root of the model dimension as per the original Transformer paper (attention is all you need)

