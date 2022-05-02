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
    config_root = os.path.join(wordl_root, "wordrl/configs/DEFAULT.yaml")
    with open(config_root, 'r') as stream:
        args = yaml.safe_load(stream)
    if len(sys.argv) > 1:
        new_config = os.path.join(
            wordl_root, f"wordrl/configs/{sys.argv[1]}.yaml")
        with open(new_config, 'r') as stream:
            new_args = yaml.safe_load(stream)
        args = wdl.setup.merge_dicts(args, new_args)
    wdl.training.run_experiment(args)
