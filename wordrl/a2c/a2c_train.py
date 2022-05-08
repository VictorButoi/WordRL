"""Advantage Actor Critic (A2C)"""
from dataclasses import dataclass
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torch

import wandb

# wordrl imports
import wordrl as wdl

AVAIL_GPUS = min(1, torch.cuda.device_count())

def main(config):
    with wandb.init(project='wordle-solver'):
        wandb.config.update(config)
        wandb.run.name = config["experiment"]["name"]

        model = wdl.a2c.module.AdvantageActorCritic(
            network_name=config["agent"]["network"],
            gamma=config["training"]["gamma"],
            lr=config["training"]["lr"],
            batch_size=config["training"]["batch_size"],
            avg_reward_len=config["training"]["avg_reward_len"],
            n_hidden=config["agent"]["n_hidden"],
            hidden_size=config["agent"]["hidden_size"],
            entropy_beta=config["training"]["entropy_beta"],
            critic_beta=config["training"]["critic_beta"],
            epoch_len=config["training"]["epoch_len"],
            prob_play_lost_word=config["training"]["prob_play_lost_word"],
            prob_cheat=config["training"]["prob_cheat"],
            weight_decay=config["training"]["weight_decay"],
            do_render=config["experiment"]["do_render"]
        ) 
        # save checkpoints based on avg_reward
        seed_everything(config["experiment"]["seed"])
                
        model_checkpoint = ModelCheckpoint(every_n_epochs=config["training"]["checkpoint_every_n_epochs"])
        trainer = Trainer(
            gpus=AVAIL_GPUS,
            max_epochs=config["training"]["epochs"],
            enable_checkpointing=True,
            callbacks=[model_checkpoint],
            resume_from_checkpoint=None
        )

        trainer.fit(model)


def train_func(config):
    main(config)
