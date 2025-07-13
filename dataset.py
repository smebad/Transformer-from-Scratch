import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):

  def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
    super().__init__()
    self.seq_len = seq_len # sequence length
    self.ds = ds # dataset
    self.tokenizer_src = tokenizer_src # source tokenizer
    self.tokenizer_tgt = tokenizer_tgt # target tokenizer
    self.src_lang = src_lang # source language
    self.tgt_lang = tgt_lang # target language
    
    self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) # start of sequence token for the source language
    self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64) # end of sequence token
    self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64) # padding token  

  def __len__(self):
    return len(self.ds) # return the length of the dataset 
  
  def __getitem__(self, index: Any) -> Any:
    src_target_pair = self.ds[index] # get the source and target pair at the given index
    src_text = src_target_pair['translation'][self.src_lang] # get the source text in the source language
    tgt_text = src_target_pair['translation'][self.tgt_lang] # get the target text in the target language

    # encode the source and target text
    enc_input_tokens = self.tokenizer_src.encode(src_text).ids # encode the source text using the source tokenizer
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # encode the target text using the target tokenizer

    # calculate the number of padding tokens needed
    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # calculate the number of padding tokens needed for the source sequence wiht -2 because we need to add the start and end of sequence tokens
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # calculate the number of padding tokens needed for the target sequence with -1 because we need to add the start of sequence token

    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0: # if the number of padding tokens is less than 0 then raise an error
      raise ValueError("Input and target sequence are too long")
    
    # add the start and end of sequence tokens to the source text
    encoder_input = torch.cat(
      [
        self.sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
      ]
    )

    # add the start of sequence token to the target text
    decoder_input = torch.cat(
      [
        self.sos_token,
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
      ]
    )

    # add the end of sequence token to the target text
    label = torch.cat(
      [
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
      ]
    )

    # check that the sizes of the tensors are correct
    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return{
      "encoder_input": encoder_input, # seq_len
      "decoder_input": decoder_input, # seq_len
      "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
      "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) and (1, seq_len, seq_len) this is done to prevent the model from looking into the future
      "label": label,
      "src_text": src_text,
      "tgt_text": tgt_text
      }
  
  
def causal_mask(size):
  mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) # create a triangular matrix with ones on the upper triangle and zeros on the lower triangle   
  return mask == 0 # convert the triangular matrix to a boolean mask