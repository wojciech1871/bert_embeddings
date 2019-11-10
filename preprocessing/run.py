import itertools
import os

from preprocessing.embeder import Embeder
from preprocessing.models import MockModel

if __name__ == '__main__':
    corpus_dir = '../doc'
    x = Embeder(corpus_dir)
    
    #model_dir = '../models/mock_polish/'
    model = MockModel()

    contexts = ['one-mention', 'document', 'corpus']
    neighborhoods = [3, 'sentence']
    for context, neighborhood in itertools.product(contexts, neighborhoods):
        save_dir = f'../projections/bert_{context}_{str(neighborhood)}/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        embeds = x.get_embeddings(context, neighborhood, model)
        x.save_to_tensorboard(embeds, save_dir)
