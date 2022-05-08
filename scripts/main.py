# Claims imports
import wordrl as wdl

# torch imports
import torch

# misc imports
import wandb
import argparse
from datetime import date
import numpy as np
import os
import yaml
import sys


if __name__ == "__main__":
    wordl_root = wdl.filepaths.FILE_PATHS["ROOT_PATH"]
    
    config_root = os.path.join(wordl_root, "wordrl/configs/DQN_DEFAULT.yaml")
    with open(config_root, 'r') as stream:
        config = yaml.safe_load(stream)
    if len(sys.argv) > 1:
        new_config = os.path.join(
            wordl_root, f"wordrl/configs/{sys.argv[1]}.yaml")
        with open(new_config, 'r') as stream:
            new_config = yaml.safe_load(stream)
        config = wdl.setup.merge_dicts(config, new_config)
        
    training_algo = config["training"]["algorithm"]
    if training_algo == "dqn":
        wdl.dqn.dqn_train.train_func(config)
    elif training_algo == "a2c":
        wdl.a2c.a2c_train.train_func(config)
    else:
        raise ValueError("Training algo not recognized!")
    
