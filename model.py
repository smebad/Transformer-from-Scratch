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
  
class MultiHeadAttentionBlock(nn.Module):

  def __init__(self, d_model: int, h: int, dropot: float) -> None:  # Initialize the multi-head attention block with the model dimension, number of heads, and dropout rate
    super().__init__()
    self.d_model = d_model # Dimension of the model
    self.h = h # Number of attention heads
    assert d_model % h == 0, "d_model must be divisible by h" # Ensure that the model dimension is divisible by the number of heads

    self.d_k = d_model // h # formula from the original Transformer paper, dimension of each head
    self.w_q = nn.Linear(d_model, d_model) # Query layer (Wq in the original Transformer paper)
    self.w_k = nn.Linear(d_model, d_model) # Key layer (Wk in the original Transformer paper)
    self.w_v = nn.Linear(d_model, d_model) # Value layer (Wv in the original Transformer paper)

    self.w_o = nn.Linear(d_model, d_model) # Output layer (Wo in the original Transformer paper)
    self.dropout = nn.Dropout(dropot) # Dropout layer for regularization

  @staticmethod # Static method to calculate the attention scores
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.size(-1) # Get the dimension of the keys

    # (Batch, h, Seq_Len, d_k) @ (Batch, h, d_k, Seq_Len) -> (Batch, h, Seq_Len, Seq_Len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # Calculate the attention scores using the dot product of the query and key tensors, scaled by the square root of d_k
    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # Set the masked positions to a very large negative value to prevent attention to those positions
    if dropout is not None:
      attention_scores = dropout(attention_scores) # Apply dropout to the attention scores if specified

    return (attention_scores @ value), attention_scores # Return the weighted sum of the values and the attention scores


  def forward(self, q, k, v, mask=None): # Forward pass through the multi-head attention block - mask is used to prevent attention to certain positions (e.g., padding tokens)
    query = self.w_q(q) # Apply the query layer to the input tensor (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
    key = self.w_k(k) # Apply the key layer to the input tensor (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
    value = self.w_v(v) # Apply the value layer to the input tensor (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)

    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # Reshape the query tensor to (Batch, h, Seq_Len, d_k) and transpose the sequence length and head dimensions
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # Reshape the key tensor to (Batch, h, Seq_Len, d_k) and transpose the sequence length and head dimensions
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # Reshape the value tensor to (Batch, h, Seq_Len, d_k) and transpose the sequence length and head dimensions
    
    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # Calculate the attention scores and apply dropout
    
    # (Batch, h, Seq_Len, d_k) -> (Batch, Seq_Len, h * d_k) -> (Batch, Seq_Len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # Transpose the head and sequence length dimensions, then reshape the tensor to (Batch, Seq_Len, d_model)

    # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
    return self.w_o(x) # Apply the output layer to the reshaped tensor (Batch, Seq_Len, d_model)
  

class ResidualConnection(nn.Module):
  
  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer): # Forward pass through the residual connection layer
    return x + self.dropout(sublayer(self.norm(x))) # Add and norm from the original Transformer paper, apply dropout to the output of the sublayer and add it to the input tensor, then apply layer normalization
  

class EncoderBlock(nn.Module):

  def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block # Multi-head attention block for self-attention
    self.feed_forward_block = feed_forward_block # Feedforward block for the encoder layer
    self.residual_connection1 = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # Residual connection for the self-attention block and feedforward block

  def forward(self, x, src_mask=None): # src mask is used to prevent attention to certain positions (e.g., padding tokens)
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # Apply the self-attention block with residual connection
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x # Apply the feedforward block with residual connection and return the output tensor
  
class Encoder(nn.Module):

  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers # List of encoder layers
    self.norm = LayerNormalization() # Layer normalization for the final output of the encoder

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask) # Apply each encoder layer to the input tensor
    return self.norm(x) # Apply layer normalization to the final output of the encoder and return the output tensor
  
class DecoderBlock(nn.Module):

  def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block # Multi-head attention block for self-attention
    self.cross_attention_block = cross_attention_block # Multi-head attention block for cross-attention (encoder-decoder attention)
    self.feed_forward_block = feed_forward_block # Feedforward block for the decoder layer
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for i in range(3)]) # Residual connections for the self-attention block, cross-attention block, and feedforward block

  def forward(self, x, encoder_output, src_mask=None, tgt_mask=None): # src_mask is used to prevent attention to certain positions in the source sequence, tgt_mask is used to prevent attention to future positions in the target sequence
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # Apply the self-attention block with residual connection to the target sequence
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # query is the target sequence, key and value are the encoder output, apply the cross-attention block with residual connection
    x = self.residual_connections[2](x, self.feed_forward_block)
    return x
  
class Decoder(nn.Module):

  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers # List of decoder layers
    self.norm = LayerNormalization()

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, tgt_mask) # Apply each decoder layer to the input tensor
    return self.norm(x) # Apply layer normalization


    
    