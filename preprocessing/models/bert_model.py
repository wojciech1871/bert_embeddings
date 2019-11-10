

from preprocessing.models.base_model import BaseModel
from utils import *

TYPE = "mean_cat_last_4_layers"


class BertModel(BaseModel):
    def __init__(self):
        self.bert = BertBaseMultilingualEmbeddingApi()

    def process(self, words_list: 'typing.List[typing.List[str]]') -> 'typing.List[np.ndarray]':
        embeds = []
        for context in words_list:
            full = " ".join(context)
            self.bert.feed_forward(full)
            embedding = self.bert.create_sentence_embedding_(how=TYPE)
            embeds.append(embedding.cpu().detach().numpy())
        return np.array(embeds)
