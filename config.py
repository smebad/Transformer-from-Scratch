from pathlib import Path
def get_config(): # returns a dictionary of configuration parameters
  return {
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 10**-4,
    "seq_len": 350,
    "d_model": 512,
    "lang_src": "en",
    "lang_tgt": "es",
    "model_folder": "weights",
    "model_filename": "tmodel_",
    "preload": None,
    "tokenizer_file": "tokenizer_{lang}.json",
    "experiment_name": "runs/tmodle" 
    
  }

def get_weights_file_path(config, epoch: str):
  model_folder = config["model_folder"]
  model_basename = config["model_filename"]
  model_filename = f"{model_basename}{epoch}.pt"
  return str/Path('.') / model_folder / model_filename
