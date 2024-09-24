#!/usr/bin/env python

# This script converts the raw predictions of the regression models to the categorical predictions using k-means clustering.
# Note that the results of the k-means clustering are different among the runs.


import pandas as pd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans

def km():
    df = pd.read_csv(f'fig/violin.csv')
    models = ['densenet','inception']
    clusters = []
    for i in range(2):
        model = df[df['model'] == models[i]]
        kmeans = KMeans(n_clusters=4,n_init='auto')
        cluster = kmeans.fit_predict(model['pred'].values.reshape(-1, 1)) 
        clusters += [int(j)+1 for j in cluster]
    df['class'] = clusters
    df.to_csv('fig/classification.csv',index=False) 
    return

# The map between the class number assigned by k-means clustering to the grade numbers {clust:grade}:
# DenseNet: {3:2,2:3}
# Inception: {2:1,3:2,1:3}

km()
