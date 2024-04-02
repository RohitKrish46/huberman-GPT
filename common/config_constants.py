from common.utils import read_config_file
import torch
config = read_config_file()

config_hypermaters = config["hyperparameters"]
BATCH_SIZE = config_hypermaters['batch_size']
BLOCK_SIZE = config_hypermaters['block_size']
MAX_ITERS = config_hypermaters['max_iters']
EVAL_INTERVAL = config_hypermaters['eval_interval']
LEARNING_RATE = config_hypermaters['learning_rate']
EVAL_ITERS = config_hypermaters['eval_iters']
N_EMBED = config_hypermaters['n_embed']
N_HEAD = config_hypermaters['n_head']
N_LAYER = config_hypermaters['n_layer']
DROPOUT = config_hypermaters['dropout']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'