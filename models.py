# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random

from torch.nn.modules.sparse import Embedding
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    
    def __init__(self):
        raise NotImplementedError
    """
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, word_embeddings):
        super (NeuralSentimentClassifier, self).__init__()
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.emd_dim = emb_dim
        self.word_embeddings = word_embeddings
        self.v = nn.Linear(emb_dim, hidden_dim)
        self.g = nn.ReLU()
        self.w = nn.Linear(hidden_dim,output_dim)
        self._softmax = nn.Softmax()
        nn.init.uniform(self.v.weight)
    
    def eee(self, x):
        return self.embedding(x)

    def forward(self, x):
        return self._softmax(self.w(self.g(self.v(x))))
    
    def predict(self, ex_words: List[str]) -> int:

        input_data = np.zeros(self.emd_dim)
        count = 0

        for word in ex_words:
            input_data += self.word_embeddings.get_embedding(word)
            count += 1
        input_data /= count

        probs = self.forward(torch.from_numpy(input_data).float())

        if probs[1] > 0.5:
            return 1
        else:
            return 0

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return [self.predict(ex_words) for ex_words in all_ex_words]

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    
    raise NotImplementedError
    """

    # parameters for training
    epoch = 1
    batch_size = 50
    learning_rate = 0.005

    # input / layer size parameters
    input_dim = 14923
    emb_dim = len(word_embeddings.vectors[0])
    hidden_dim = 256
    output_dim = 2

    # establish model and optimizer
    criterion = nn.CrossEntropyLoss()
    model = NeuralSentimentClassifier(input_dim, emb_dim, hidden_dim, output_dim, word_embeddings)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # training
    for _ in range(epoch):

        # batching
        random.Random(1).shuffle(train_exs)
        
        train_sentences = [w.words for w in train_exs]
        train_labels = [w.label for w in train_exs]

        # onehot encoded the labels
        onehot_encoded = list()
        for value in train_labels:
            classes = [0.0 for _ in range(output_dim)]
            classes[value] = 1.0
            onehot_encoded.append(classes)
        train_labels = onehot_encoded

        for i in range(len(train_exs)):

            # averaging the word vectors
            input_data = np.zeros(emb_dim)
            count = 0
            for word in train_sentences[i]:
                #input_data[word_embeddings.word_indexer.index_of(word)] += 1
                input_data += word_embeddings.get_embedding(word)
                count += 1
            input_data /= count
            input_data = torch.from_numpy(input_data).float()

            #initialize
            optimizer.zero_grad()
            model.zero_grad()

            probs = model.forward(input_data)
            loss = torch.sum(torch.neg(torch.log(probs)).dot(torch.tensor(train_labels[i])))

            loss.backward()
            optimizer.step()

    return model

