

import os
import re
import spacy
from spacy.lang.xx import MultiLanguage
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from preprocessing.document import Document


class Embeder:
    def __init__(self, corpus_dir):
        self._data = self._load_data(corpus_dir)

    def _load_data(self, corpus_dir):
        data = {}
        #nlp = spacy.load("xx")
        nlp = MultiLanguage()
        for root, _, files in os.walk(corpus_dir):
            if not files:
                continue

            for doc_name in tqdm(files, desc=f'Loading corpus from {root}...'):
                doc_path = os.path.join(root, doc_name)
                doc = Document.from_file(doc_path, nlp)
                data[doc_path] = doc
        return data

    def get_embeddings(self, context, neighborhood, model):
        assert context in ['one-mention', 'document', 'corpus']
        embeddings = defaultdict(list)
        for _, doc in tqdm(self._data.items(), desc=f'Computing embeddings ({model.__class__.__name__}, {context}, {str(neighborhood)})...'):
            neighborhood_dict = doc.get_neighbors(neighborhood)  # return words in specified neighbourhood
            for (person, category), doc_neighbors in neighborhood_dict.items():
                one_mention_embeds = np.vstack(model.process(doc_neighbors))
                # one_mention_embeds = np.vstack([model.process(one_mention_neighbors) for one_mention_neighbors in doc_neighbors])
                if context == 'one-mention':
                    embeddings[(person, category)].append(one_mention_embeds)
                else:
                    doc_embeds = np.mean(one_mention_embeds, axis=0, keepdims=True)
                    embeddings[(person, category)].append(doc_embeds)  # Becomes doc embeds

        for (person, category), doc_embeds in embeddings.items():
            embeddings_array = np.concatenate(doc_embeds, axis=0)
            if context == 'corpus':
                embeddings_array = np.mean(embeddings_array, axis=0, keepdims=True)
            embeddings[(person, category)] = embeddings_array
        return embeddings

    def save_to_tensorboard(self, embeddings, save_dir):
        metadata, embeds = [], []
        for (name, category), person_embeds in embeddings.items():
            metadata += [f'{category}: {name}'] * person_embeds.shape[0]
            embeds.append(person_embeds)
        embeds = np.concatenate(embeds, axis=0)

        writer = SummaryWriter(log_dir=save_dir)
        writer.add_embedding(embeds, metadata=metadata)
        writer.close()
