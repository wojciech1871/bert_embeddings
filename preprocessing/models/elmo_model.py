from .base_model import BaseModel
#from elmoformanylangs import Embedder
import numpy as np
import logging 


class ElmoModel(BaseModel):
    def __init__(self, path_to_model):
        #self._embedder = Embedder(path_to_model)
        pass

    def process(self, sentences):
        return [np.mean(embeds, axis=0) for embeds in self._embedder.sents2elmo(sentences)]
