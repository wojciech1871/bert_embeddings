from .base_model import BaseModel

import numpy as np


class MockModel(BaseModel):
    def process(self, sentences):
        return [np.random.random(size=(8)) for _ in range(len(sentences))]
