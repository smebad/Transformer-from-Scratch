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

class ProjectionLayer(nn.Module):

  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.projection = nn.Linear(d_model, vocab_size) # Linear layer to project the output of the decoder to the vocabulary size
    
  def forward(self, x):
    # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, vocab_size)
    return torch.log_softmax(self.projection(x), dim = -1) # Apply the projection layer and log softmax to the output tensor, return the log probabilities of the vocabulary tokens
  
class Transformer(nn.Module):

  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder # Encoder for the input sequence
    self.decoder = decoder # Decoder for the target sequence
    self.src_embed = src_embed # Input embeddings for the source sequence
    self.tgt_embed = tgt_embed # Input embeddings for the target sequence
    self.src_pos = src_pos # Positional encoding for the source sequence
    self.tgt_pos = tgt_pos # Positional encoding for the target sequence
    self.projection_layer = projection_layer # Projection layer to project the output of the decoder to the vocabulary size

  def encode(self, src, src_mask):
    src = self.src_embed(src) # Apply the source embeddings to the input sequence
    src = self.src_pos(src) # Apply positional encoding to the source embeddings
    return self.encoder(src, src_mask) # Encode the source sequence using the encoder
  
  def decode(self, tgt, encoder_output, src_mask, tgt_mask):
    tgt = self.tgt_embed(tgt) # Apply the target embeddings to the target sequence
    tgt = self.tgt_pos(tgt) # Apply positional encoding to the target embeddings
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask) # Decode the target sequence using the decoder
  
  def project(self, x):
    return self.projection_layer(x) # Project the output of the decoder to the vocabulary size using the projection layer
  
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer: # Function to build the Transformer model
  
  # embedding layers
  src_embed = InputEmbeddings(d_model, src_vocab_size) # Create the source embeddings layer
  tgt_embed = InputEmbeddings(d_model, tgt_vocab_size) # Create the target embeddings layer

  # positional encoding layers
  src_pos = PositionalEncoding(d_model, src_seq_len, dropout) # Create the positional encoding layer for the source sequence
  tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # Create the positional encoding layer for the target sequence

  # encoder blocks
  encoder_blocks = [] # List to hold the encoder blocks
  for i in range(N):
    encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Create the multi-head attention block for self-attention
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # Create the feedforward block for the encoder layer
    encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout) # Create the encoder block with self-attention and feedforward blocks
    encoder_blocks.append(encoder_block) # Append the encoder block to the list

  # decoder blocks
  decoder_blocks = [] 
  for i in range(N):
    decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Create the multi-head attention block for cross-attention (encoder-decoder attention)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # Create the feedforward block for the decoder layer
    decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block) # Append the decoder block to the list

  # creating the encoder and decoder
  encoder = Encoder(nn.ModuleList(encoder_blocks)) # Create the encoder with the list of encoder blocks
  decoder = Decoder(nn.ModuleList(decoder_blocks)) # Create the decoder with the list of decoder blocks

  # projection layer (output layer)
  projection_layer = ProjectionLayer(d_model, tgt_vocab_size) # Create the projection layer to project the output of the decoder to the vocabulary size

  # creating the transformer model
  transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer) # Create the Transformer model with the encoder, decoder, embeddings, positional encodings, and projection layer

  # initialize the weights of the model (Xavier initialization)
  for p in transformer.parameters():
    if p.dim() > 1: # Apply Xavier initialization to weights
      nn.init.xavier_uniform_(p) # Initialize the weights of the model using Xavier uniform initialization

  return transformer # Return the Transformer model
