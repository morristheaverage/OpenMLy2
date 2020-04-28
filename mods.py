# Auxiliary program to compare various modules
import pandas as pd
import numpy as np
from collections import defaultdict

stuinfo = pd.read_csv('studentInfo.csv')
modules = pd.read_csv('courses.csv')
print('Loaded data')

# How many students do each module
studentMods = defaultdict(int)

for _, row in stuinfo.iterrows():
    # Course
    studentMods[row['code_module']] += 1
    # Year
    studentMods[row['code_presentation'][3]] += 1
    # Month
    studentMods[row['code_presentation'][4]] += 1
    

