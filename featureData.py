# Auxiliary program to generate data about features
# Aim is to inform encoding strategies

# First example is using k-clustering on the region of a student
# Read in studentInfo.csv to stuInfo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

stuInfo = pd.read_csv("studentInfo.csv")

N = 3
kmeans = KMeans(n_clusters=N, random_state=42)

# Create a dictionary to map final_result to a unique value
resDim = {result:i for i, result in enumerate(np.unique(stuInfo["final_result"]))}

# Build a vector for each region
regions = {region:i for i, region in enumerate(np.unique(stuInfo["region"]))}
vectors = [[0 for _ in resDim] for _ in regions]
for reg, res in zip(stuInfo["region"], stuInfo["final_result"]):
    vectors[regions[reg]][resDim[res]] += 1

# Normalise vectors
for vector in vectors:
    total = sum(vector)
    for i in range(len(vector)):
        vector[i] /= total

# Perform clustering
kmeans.fit(vectors)


# View results
groups = {i:[] for i in range(N)}
for label, reg in zip(kmeans.labels_, regions.keys()):
    groups[label].append(reg)

for g in groups.values():
    print(g)

def distance(reg1, reg2):
    coord1 = vectors[regions[reg1]]
    coord2 = vectors[regions[reg2]]
    return np.sqrt(sum([(c1 - c2)**2 for c1, c2 in zip(coord1, coord2)]))

def orderAround(reg):
    A = [k for k in regions.keys()]
    A.sort(key=lambda k: distance(reg, k))
    return A
