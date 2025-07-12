import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset

from dataset import BilingualDataset, casual_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
  model.eval() # set the model to evaluation mode in PyTorch
  count = 0

  source_texts = [] # list to store the source text
  expected = [] # list to store the expected translation
  predicted = [] #  predicted translation

  # Size of the control window (just use a default value)
  console_width = 80 # 80 characters

  with torch.no_grad(): # disable gradient calculation
    for batch in validation_ds:
      count += 1
      encoder_input = batch['encoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)

      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation" # check that the batch size is 1
      


def get_all_sentences(ds, lang):
  for item in ds:
    yield item['translation'][lang]  # Assuming 'translation' is a dict with language keys
def get_or_build_tokenizer(config, ds, lang):
  # config['tokenizer_file'] is a path to the tokenizer file
  tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))

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

  train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len']) # create a dataset object for the training set
  val_ds = BilingualDataset(val_ds_size, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len']) # create a dataset object for the validation set

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids # encode the source text using the source tokenizer
    tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids # encode the target text using the target tokenizer
    max_len_src = max(max_len_src, len(src_ids)) # update the maximum length of the source sequence
    max_len_tgt = max(max_len_tgt, len(tgt_ids)) # update the maximum length of the target sequence

  print(f"Max length of source sentence: {max_len_src}")
  print(f"Max length of target sentence: {max_len_tgt}")

  # create dataloaders
  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True) # create a dataloader for the training set
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True) # create a dataloader for the validation set with batch size of 1

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt 

def get_model(config, vocab_src_len, vocab_tgt_len): # vocab_src_len and vocab_tgt_len are the lengths of the source and target vocabularies
  model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
  return model 

def train_model(config):
  # defining the devcie
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Device: {device}")

  Path(config['model_folder']).mkdir(parents=True, exist_ok=True) # create the model folder if it does not exist

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config) # get the dataloaders and tokenizers
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device) # get the model and move it to the device

  # Tensorboard - for logging training metrics
  writer = SummaryWriter(config['experiment_name'])

  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9) # Adam optimizer with learning rate of 10^-4 and epsilon of 10^-9

  initial_epoch = 0
  global_step = 0
  if config['preload']:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f'Preloading model from {model_filename}')
    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']

  loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # define the loss function with ignore index of the padding token and label smoothing of 0.1

  # Training loop
  for epoch in range(initial_epoch, config['num_epochs']):
    model.train() # set the model to training mode
    batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch: 02d}') # create a tqdm progress bar
    for batch in batch_iterator:

      # Moving the tensors to the device
      encoder_input = batch['encoder_input'].to(device) # move the encoder input to the device (B, seq_len)
      decoder_input = batch['decoder_input'].to(device) # move the decoder input to the device (B, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

      # Running the tensors through the transformer
      encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model) - encoder output
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model) - decoder output
      proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size) - output of the projection layer

      label = batch['label'].to(device) # move the label to the device (B, seq_len)

      # Transform (B, Seq_len, tgt_vocab_size) to (B * Seq_len, tgt_vocab_size)
      loss = loss_function(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) # calculate the loss
      batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'}) # update the tqdm progress bar with the loss

      # logging the loss to tensorboard
      writer.add_scalar('train_loss', loss.item(), global_step) # add the loss to tensorboard
      writer.flush() # flush the tensorboard writer

      # Backpropagating the loss
      loss.backward()

      # Update the weights
      optimizer.step()
      optimizer.zero_grad()

      global_step += 1 # increment the global step counter

      # Saving the model at the end of each epoch
      model_filename = get_weights_file_path(config, f'{epoch:02d}')
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
      }, model_filename)

# Running the main function
if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  config = get_config()
  train_model(config)