import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

class BertBaseMultilingualEmbeddingApi:
    
    def __init__(self, model_name="bert-base-multilingual-cased", cuda=True):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.tokens_tensor = None
        self.segments_tensor = None
        
        self.encoded_layers_ = None
        self.token_embeddings_ = None
        
    def _tokenize_text(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        
        self.tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        self.segments_tensor = torch.tensor([segments_ids]).to(self.device)
    
    def _evaluate(self):
        with torch.no_grad():
            encoded_layers, _ = self.model(self.tokens_tensor, self.segments_tensor)
        self.encoded_layers_ = encoded_layers
    
    def _generate_token_embeddings(self, batch_i=0):
        """
        Convert the hidden state embeddings into single token vectors
        Holds the list of 12 layer embeddings for each token
        Will have the shape: [# tokens, # layers, # features]
        """
        token_embeddings = [] 
        # For each token in the sentence...
        for token_i in range(len(self.encoded_layers_[-1][batch_i])):
            # Holds 12 layers of hidden states for each token 
            hidden_layers = [] 
            # For each of the 12 layers...
            for layer_i in range(len(self.encoded_layers_)):
            # Lookup the vector for `token_i` in `layer_i`
                vec = self.encoded_layers_[layer_i][batch_i][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        self.token_embeddings_ = token_embeddings
    
    def feed_forward(self, sentence):
        self._tokenize_text(sentence)
        self._evaluate()
        self._generate_token_embeddings()
    
    def create_word_embedding_(self, how="cat_last_4"):
        if how == "cat_last_4":
            return [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in self.token_embeddings_]
        elif how == "sum_last_4":
            return [torch.sum(torch.stack(layer)[-4:], 0) for layer in self.token_embeddings_]
        else:
            print("Redefine `how` parameter")
        
    def create_sentence_embedding_(self, how="mean_last_layer"):
        if how == "mean_last_layer":
            return torch.mean(self.encoded_layers_[-1], 1).squeeze()
        elif how == "mean_cat_last_4_layers":
            return torch.mean(torch.cat((self.encoded_layers_[-1], self.encoded_layers_[-2], self.encoded_layers_[-3], self.encoded_layers_[-4]), 2), 1).squeeze()
            #return torch.mean(torch.stack(self.create_word_embedding_("cat_last_4")), 0)
        elif how == "mean_sum_last_4_layers": 
            return torch.mean(torch.sum(torch.stack(self.encoded_layers_[-4:]), 0), 1).squeeze()
            #return torch.mean(torch.stack(self.create_word_embedding_("sum_last_4")), 0)
        else:
            print("Redefine `how` parameter")
            
    def print_dimensions_(self):
        print ("Number of layers:", len(self.encoded_layers_))
        layer_i = 0
        print ("Number of batches:", len(self.encoded_layers_[layer_i]))
        batch_i = 0
        print ("Number of tokens:", len(self.encoded_layers_[layer_i][batch_i]))
        token_i = 0
        print ("Number of hidden units:", len(self.encoded_layers_[layer_i][batch_i][token_i]))
        
    def plot_embedding_hist(self, vec):
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 8))
        sns.distplot(vec)
        plt.show()
