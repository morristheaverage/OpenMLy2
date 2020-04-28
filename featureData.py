# Auxiliary program to generate data about features
# Aim is to inform encoding strategies

# First example is using k-clustering on the region of a student
# Read in studentInfo.csv to stuInfo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

stuInfo = pd.read_csv("studentInfo.csv")

print("Loaded data")

# The first approach uses kmeans clustering to divide
# the regions into N groups based on student results
N = 2
kmeans = KMeans(n_clusters=N, random_state=42)

# Create a dictionary to map final_result to a unique value
resDim = {result:i for i, result in enumerate(np.unique(stuInfo["final_result"]))}
print("Built result dictionary")

# Build a vector for each region
regions = {region:i for i, region in enumerate(np.unique(stuInfo["region"]))}
vectors = [[0 for _ in resDim] for _ in regions]
print("Built region vectors")

for reg, res in zip(stuInfo["region"], stuInfo["final_result"]):
    vectors[regions[reg]][resDim[res]] += 1

# Normalise vectors
for vector in vectors:
    total = sum(vector)
    for i in range(len(vector)):
        vector[i] /= total
print("Normalised vectors")

# Perform clustering
kmeans.fit(vectors)
print("Clustering done")

# Second approach will score each region
# Init container to tally student results per region
#       Dictionary rt=regiongTallies key=region  value=tally of each result
rT = {region:defaultdict(int) for region in np.unique(stuInfo['region'])}

# Create counts
for reg, res in zip(stuInfo['region'], stuInfo['final_result']):
    rT[reg][res] += 1

# Create proportional data
for reg in rT:
    total = sum(rT[reg].values())
    print(reg, total)
    for res in rT[reg]:
        rT[reg][res] /= total

# Normalise data
for res in np.unique(stuInfo['final_result']):
    mean = sum(rT[reg][res] for reg in stuInfo['region'])/len(rT)
    var = sum((rT[reg][res] - mean)**2 for reg in stuInfo['region'])/len(rT)
    print(res, mean, var)

# Helper function to score regions
def score(data, reg):
    # Assign coefficients to weight each outcome
    coef = defaultdict(int)
    coef['Fail'] = 0
    coef['Withdrawn'] = 0
    coef['Pass'] = 1
    coef['Distinction'] = 2

    # Produce score
    inputs = ['Fail', 'Withdrawn', 'Pass', 'Distinction']
    return sum(data[reg][x] * coef[x] for x in inputs)

# Score regions
rs = {reg: score(rT, reg) for reg in np.unique(stuInfo['region'])}


# View results
groups = {i:[] for i in range(N)}
for label, reg in zip(kmeans.labels_, regions.keys()):
    groups[label].append(reg)

print('First results')
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
