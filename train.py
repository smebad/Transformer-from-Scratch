import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
  for item in ds:
    yield item['translation'][lang]  # Assuming 'translation' is a dict with language keys
def get_or_build_tokenizer(config, ds, lang):
  # config['tokenizer_file'] is a path to the tokenizer file
  tokenizer_path = Path(config['tokenizer_file']).format(lang)
  if not Path.exists(tokenizer_path): # if the tokenizer file does not exist
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # create a new tokenizer
    tokenizer.pre_tokenizer = Whitespace() # set the pre-tokenizer to split on whitespace
    trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2) # create a trainer with special tokens 
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) # train the tokenizer on all sentences in the dataset
    tokenizer.save(str(tokenizer_path)) # save the tokenizer to the file
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path)) # load the tokenizer from the file
  return tokenizer

def get_ds(config):
  ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train') # load the dataset from the opus_books library with the specified source and target languages

  # Building tokenizers for source and target languages
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  # splitting the dataset
  train_ds_size = int(0.9 * len(ds_raw)) # 90% of the dataset size
  val_ds_size = len(ds_raw) - train_ds_size # the rest of the dataset size
  train_ds_raw, val_ds_size = random_split(ds_raw, [train_ds_size, val_ds_size]) # split the dataset into training and validation sets