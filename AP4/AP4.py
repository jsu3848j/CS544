import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from collections import Counter

names=['area', 'perimeter', 'compactness', 'length of kernel', 'width of kernel', 'asymmetry coefficient', 'length of kernel groove', 'variety of wheat']
data1 = pd.read_table('seeds_dataset.txt', delimiter='\t', names=names)

feature_cols = ['area', 'perimeter', 'compactness', 'length of kernel', 'width of kernel', 'asymmetry coefficient', 'length of kernel groove']
data2 = data1[feature_cols]
variety = data1['variety of wheat']

k = 3

kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(data2)
cvar = [[] for _ in range(k)]
total = 0

for i in range(0,k):
    c = np.where(clusters==i)[0].tolist()
    cv = variety[c]
    count = Counter(cv).most_common(1)
    cvar[i] = count[0][0]
    total = total + count[0][1]

accuracy = float(total)/len(variety)
print "Accuracy: %1.2f" %(accuracy * 100.00)
centriods = kmeans.cluster_centers_.tolist()

for i in range(0,k):
    centriods[i].append(cvar[i])
    print zip(names,centriods[i])
    