from typing import List

import numpy as np
import torch
from torch import nn
from embeddings import GloveEmbedding
import spacy

"""
Options for different embedding conversion matrices into the softmax: 

"original_random" : an entirely random matrix of len(wordlist) x (26 * 5) values. Each row is guaranteed to be unique in order and value 
"original": The original embedding conversion matrix. len(wordlist) x (26 * 5) dimension, where each row denotes the location of 
each letter with 1 or 0 

"gloVe": The embedding conversion matrix using G representations
"spacy": Embedding using spaCy

"matrix_width" : the length of the embedding before passing into actor or critic, only used in "original_random" and "glove". 
26*5 is the size of the "original", and 300 is safe for glove for sure


"num_actor_layers", "num_critic_layers": number of linear layers for actor and critic. Final linear layer has no ReLU. 
Must be at least 1

"glove_dataset" : dataset used to make glove embeddings

"""


class SumChars(nn.Module):
    def __init__(self, obs_size: int, word_list: List[str], n_hidden: int = 1, hidden_size: int = 256, embedding_matrix = "original", matrix_width = 26*5, num_actor_layers = 1, num_critic_layers = 1, glove_dataset = "common_crawl_840"):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()

        embedding_width = None 

        self.words = None #word representation matrix 

        if embedding_matrix == "original":
            embedding_width = 26*5 #for each of 5 letters, 1-26


            #column vector = word
            word_array = np.zeros((embedding_width, len(word_list)))
            for i, word in enumerate(word_list):
                for j, c in enumerate(word): #j = letter order in word
                    word_array[j*26 + (ord(c) - ord('A')), i] = 1
            self.words = torch.Tensor(word_array)

        elif embedding_matrix == "original_random":
            embedding_width = matrix_width
            self.words = torch.Tensor(np.random(embedding_width, len(word_list)))

        elif embedding_matrix == "glove":
            embedding_width = matrix_width
            g = GloveEmbedding(glove_dataset, d_emb = glove_demb)
            word_emb_array = np.array([g.emb(w) for w in word_list])
            self.words = torch.Tensor(word_emb_array)

        elif embedding_matrix == "spacy":
            nlp = spacy.load("en_core_web_md")
            self.words = np.concatenate([nlp(w) for w in word_list])
            embedding_width = len(self.words[0]) # should be 300 

            

        else: 
            raise ValueError("Invalid embedding matrix ID")

        layers = [
            nn.Linear(obs_size, hidden_size),
            nn.ReLU()
            ]

        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, embedding_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)


        #Create actor 

        actor_layers = []

        for i in range(num_actor_layers - 1):
            actor_layers.append(nn.Linear(embedding_width, embedding_width))
            actor_layers.append(nn.ReLU())

        actor_layers.append(nn.Linear(embedding_width, embedding_width))
        self.actor_head = nn.Sequential(*actor_layers)

        #Create critic

        critic_layers = []

        for i in range(num_critic_layers - 1):
            critic_layers.append(nn.Linear(embedding_width, embedding_width))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(embedding_width, 1))

        self.critic_head = nn.Sequential(*critic_layers)

    def forward(self, x):
        y = self.f0(x.float()) #turn to float + use nn 
        a = torch.log_softmax(
            torch.tensordot(self.actor_head(y), #actor_head = linear network, width = embedding_width 
                            self.words.to(self.get_device(y)), #dot product, words matrix he makes; embedding_width x wordlist 
                            dims=((1,), (0,))),
            dim=-1)
        c = self.critic_head(y)
        return a, c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index
