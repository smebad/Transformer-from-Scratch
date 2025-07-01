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
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_Len: int, dropout: float) -> None: # Initialize the positional encoding layer 
    super().__init__()
    self.d_model = d_model # Dimension of the model
    self.seq_Len = seq_Len # Length of the input sequence
    self.dropout = nn.Dropout(dropout) # Dropout layer for regularization

    # Create a matrix of shape (seq_Len, d_model) to hold the positional encodings
    pe = torch.zeros(seq_Len, d_model) # Initialize the positional encoding matrix with zeros

    # Create a vector of shape (Seq_Len, 1) to hold the position indices
    position = torch.arange(0, seq_Len, dtype=torch.float).unsqueeze(1) # Initialize the position indices vector with the range of values from 0 to seq_Len)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Calculate the division term for the positional encoding formula

    # Apply the positional encoding formula to even and odd indices
    pe[:, 0::2] = torch.sin(position * div_term) # Apply sine function to even indices
    pe[:, 1::2] = torch.cos(position * div_term) # Apply cosine function to odd indices

    pe = pe.unsqueeze(0) # Add a batch dimension to the positional encoding matrix (1, seq_Len, d_model)

    self.register_buffer('pe', pe) # Register the positional encoding matrix as a buffer so that it is not considered a model parameter

  def forward(self, x): # Forward pass through the positional encoding layer
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Add the positional encodings to the input embeddings and set the positional encodings as non-learnable
    return self.dropout(x) # Apply dropout to the output of the positional encoding layer

class LayerNormalization(nn.Module):
  def __init__(self, eps: float = 1e-6): # Initialize the layer normalization layer with a small epsilon value of 10-6 to avoid division by zero
    super().__init__()
    self.eps = eps  # Small value to avoid division by zero
    self.alpha = nn.Parameter(torch.ones(1)) # Scale parameter for normalization (multplication)
    self.bias = nn.Parameter(torch.zeros(1)) # Bias parameter for normalization (addition)

  def forward(self, x): # Forward pass through the layer normalization layer
    mean = x.mean(dim = -1, keepdim = True) # Calculate the mean of the input tensor along the last dimension
    std = x.std(dim = -1, keepdim = True) # Calculate the standard deviation of the input tensor along the last dimension)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias # Normalize the input tensor and apply the scale and bias parameters, formula from the original Transformer paper

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:  # Initialize the feedforward block with the model dimension, feedforward dimension, and dropout rate
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff) # First linear layer to project the input to the feedforward dimension (W1 and b1 in the original Transformer paper)
    self.dropout = nn.Dropout(dropout) # Dropout layer for regularization
    self.linear2 = nn.Linear(d_ff, d_model) # Second linear layer to project the output back to the model dimension (W2 and b2 in the original Transformer paper)

  def forward(self, x): # Forward pass through the feedforward block
    # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_ff) -> (Batch, Seq_Len, d_model). Formula from the original Transformer paper
    return self.linear2(self.dropout(torch.relu(self.linear1(x)))) # Apply the first linear layer, ReLU activation, dropout, and the second linear layer in sequence
    