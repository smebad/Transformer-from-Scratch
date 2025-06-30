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
  
  def positional_encoding(self, d_model: int, seq_Len: int, dropout: float) -> None: # Method to create positional encodings
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
