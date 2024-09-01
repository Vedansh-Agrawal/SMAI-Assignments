import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import pandas as pd
from sklearn import metrics, neighbors, model_selection, tree, preprocessing, compose
import heapq
import time
import seaborn
from functools import reduce
import sys


class kNN():
    def __init__(self):
        self.pred = []
        self.report = ''
        self.metrics = {'Accuracy': 0.0, 'F1-score':0.0, 'Recall':0.0, 'Precision':0.0}

    def euclid(self, embed1, embed2):
        return np.sqrt(np.sum((embed1 - embed2)*(embed1 - embed2), axis = 1)) # Computes the Euclidean distance

    def manhattan(self, embed1, embed2):
        return np.sum(np.abs(embed1 - embed2), axis = 1) # Computes the Manhattan distance

    def cosine(self, embed1, embed2):
        return 1 - np.dot(np.squeeze(embed1), np.squeeze(embed2).T)/(np.linalg.norm(embed1)*np.linalg.norm(embed2, axis = 1)) # Computes the Cosine distance
  

    def split(self, i, data):# i is the training mark in percent of data we want as training data
        i = i*len(data)
        training_data = data[:int(i), :] # splitting training data
        validation_data = data[int(i):, :] # splitting testing data

        return training_data, validation_data

    def prediction(self, test_sample, training_data, encoder, dist_metric, k=1):
        embed_col = 1 if encoder == 'ResNet' else 2

        test_embed = test_sample[embed_col] # Extracting embeddings, given the encoder
        training_embeds = np.vstack(training_data[:, embed_col])
        train_class = np.vstack(training_data[:, 3])
        

        dist = dist_metric(test_embed , training_embeds)
        dist_dupl = dist # Computing distances with every vector in test set.
        heapq.heapify(list(dist_dupl)) # Sorting the accuracies using a heap
        
        k_nearest = heapq.nsmallest(k, dist_dupl)
        
            
        classes = [train_class[np.where(dist == k_nearest[i])] for i in range(len(k_nearest))]
        
        class_count = Counter(np.squeeze(np.vstack(classes).T))

        self.pred.append(max(class_count, key= lambda x: class_count[x]))
        

    def inference(self, test_data, training_data, encoder, dist_metric, k=1):
        self.pred = []
        for i in range(len(test_data)):
            self.prediction(test_data[i, :], training_data, encoder, dist_metric, k)

        self.report = metrics.classification_report(test_data[:, 3], self.pred, zero_division = 0)
        self.metrics['Accuracy'] = metrics.accuracy_score(test_data[:, 3], self.pred)
        self.metrics['F1-score'] = metrics.f1_score(test_data[:, 3], self.pred, average = 'micro')
        self.metrics['Recall'] = metrics.recall_score(test_data[:, 3], self.pred, average = 'micro')
        self.metrics['Precision'] = metrics.precision_score(test_data[:, 3], self.pred, average = 'micro')
        
        return metrics.accuracy_score(test_data[:, 3], self.pred)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python process_np_file.py <input_np_file.npy>")
        sys.exit(1)

    input_np_file = sys.argv[1]
    validation_data = np.load(input_np_file, allow_pickle=True)


    # unpickling the data
    with open('data.npy', 'rb') as f:
        training_data = pickle.load(f)


    model = kNN()
    model.inference(validation_data, training_data, 'VIT', model.cosine, 9)
    print(model.report)
    print(model.metrics)

