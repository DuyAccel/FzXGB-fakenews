#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.feature_selection import SelectKBest, chi2

import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[2]:


class Dataset:
    def __init__(self, folder_path):
        """
        
        Initializing neccessary atribute and reading LIAR dataset from the folder directory.
        LIAR dataset includes:
            train.tsv - training set.
            valid.tsv - validation set.
            test.tsv - testing set.
        **Because this experiment do not need valid set, I will concatenate it with training set.**

        Freatures of this dataset includes:
            ['ID', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'affiliation',
             'barely-true', 'false', 'half-true', 'mostly-true', 'pant-on-fire', 'context'] 
             
        - [statement]      is the text we need to categorize.
        - [label]          is target.
        - [Others feature] is meta-data of the statement.
        - [ID]             is the id of the statement (not neccessary).
        
        ---
        Inputs :
            folder_path: A string contain path to dataset.
        Output :
            _
            
        """
        self.train = None
        self.test = None
        
        self.y_train = None
        self.y_test = None
        
        self.features = ['ID', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'affiliation',
         'barely-true', 'false', 'half-true', 'mostly-true', 'pant-on-fire', 'context']
        self.train = pd.concat([pd.read_csv(folder_path+'/train.tsv', delimiter='\t', names=self.features, quoting=3),
                                pd.read_csv(folder_path+'/valid.tsv', delimiter='\t', names=self.features, quoting=3)],
                                ignore_index=True)
        self.test = pd.read_csv(folder_path+'/test.tsv', delimiter='\t', names=self.features, quoting=3)
        self.y_train = self.train['label']
        self.y_test = self.test['label']
        self.target = 'label'
        self.liar_preprocess()

    def subtract_current_credit(self, row):
        """
        
        Subtract the current label from the credit history of current statement.
            
        """
        label = row['label'] 
        try:
            row[label] -= 1  
        except:
            pass
        return row
            
    def liar_preprocess(self):
        """
    
        Preproces steps of LIAR dataset which is include:
            * Define which are text features
            * Define which are numeric features (credit history)
            * Fill the blanks (missing cell) of text features with word [unknow]
            * Prevent data leakage from numeric features
                According to dataset author: 
                    "Credit history include the count of the current statement, 
                    it is important to subtract the current label from the credit history when using this 
                    meta data vector in prediction experiments."

        """
        self.text_features = ['statement', 'subject', 'speaker', 'job', 'state', 'affiliation', 'context']
        self.num_features = ['barely-true', 'false', 'half-true', 'mostly-true', 'pant-on-fire']
        self.train[self.text_features] = self.train[self.text_features].fillna("unknow").astype(str)
        self.test[self.text_features] = self.test[self.text_features].fillna("unknow").astype(str)

        self.train = self.train.apply(self.subtract_current_credit, axis=1)
        self.test = self.test.apply(self.subtract_current_credit, axis=1)
        


# ## Tiền xử lý dữ liệu

# In[3]:


STOPWORDS = set(stopwords.words('english'))
PUNCT_TRANS = str.maketrans('', '', string.punctuation)

def text_preprocess(text):
    """
    
    Clean statement feature.
    
    """
    text = text.lower()
    text = text.translate(PUNCT_TRANS)
    text = ' '.join([word for word in word_tokenize(text) if word not in STOPWORDS])
    return text

def context_preprocess(text):
    """
    
    Clean context feature.
    
    """
    text = text.lower()
    text = re.sub('e mail|e-mail|email|mailer','mail', text)
    text = re.sub('television','tv', text)
    text = re.sub('website','web', text)
    text = text.translate(PUNCT_TRANS)
    text = ' '.join([word for word in word_tokenize(text) if word not in STOPWORDS])
    return text
    
def subject_preprocess(text):
    """
    
    Clean subject feature.
    
    """
    text = text.lower()
    text = ' '.join(text.split(','))
    return text

def job_preprocess(text):
    """
    
    Clean job feature.
    
    """
    text = text.lower()
    text = text.translate(PUNCT_TRANS)
    return text


# ## Feature Extraction

# In[4]:


class Fuzzifier():
    def __init__(self):
        """
        
        A model calculate centroids of training set (according to its labels).
        After fitting with training data, this model can create fuzzy sets (membership values matrixs) for new data points
        
        """
        self.centroids = None
        
    def __calculate_centroids(self, data, label, targets):
        """
        
        Calculate centroids of data
        ---
        Inputs :
            data : A matrix prepresent list of data points.
            label : A list contains labels of each row in data.
            targets : A list contains 
        Output :
            A matrix. Each column of matrix is a vector prepresent a centroid of data.
    
        """
        centroids = np.zeros((len(targets), data.shape[1])) 
        for i, target in enumerate(targets):
            cluster_data = data[label == target]
            centroids[i, :] = np.mean(cluster_data, axis=0) 
        return centroids
    def __estimateDistances(self, x, V):
        """
        
        Estimating Euclid distances from x to V.
        ---
        Inputs :
            x : A vector prepresent a object (row) in dataset.
            v : A matrix prepresent list of centroids.
        Output :
            A list of Euclid distances from x to V.
    
        """
        
        distances = np.empty(len(V))
        for i in range(len(V)):
            distances[i] = np.power(norm(x - V[i]), 2)
        return distances
    
    def __getMembershipValue(self, X, V, c, m = 4):
        """
        
        Calculate membership values of data (X) to centroids (V).
        ---
        Inputs :
            X : A matrix prepresent list of data points.
            V : A matrix prepresent list of centroids.
            c : Number of centroids.
            m : Fuzzy parameter
        Output :
            A matrix prepresent membership values of data (X) to centroids (V).
    
        """
        
        U = np.empty((X.shape[0], c))
        p = float(2/(1-m))
        for i in range(X.shape[0]):
            x = X[i].toarray()
            distances = self.__estimateDistances(x, V)
            
            if (distances.min() == 0.0):
                for j in range(c):
                    U[i][j] = 0
                U[i][np.argmin(distances)] = 1
                continue    
                
            U[i] = np.power(distances, p)
            Sum = np.sum(U[i])
            U[i] = U[i]/Sum
        return U

    def fit(self, X, y, n_class):
        """
        
        Initialize centroids of the giving data.
        ---
        Inputs :
            X : A matrix prepresent list of data points.
            y : A list contains labels of each row in data.
            n_class : The number of labels.
        Output :
            -
        """
        self.centroids = self.__calculate_centroids(X, y, range(n_class))
        
    def predict(self, X):
        """
        Calculate fuzzy set (membership values matrix) of the giving data.
        ---
        Inputs :
            X : A matrix prepresent list of data points.
        Output :
            A matrix prepresent membership values of data (X) according to model's centroids.

        """
        return self.__getMembershipValue(X=X, V=self.centroids, c=2)

