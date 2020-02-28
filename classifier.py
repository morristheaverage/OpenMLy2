import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numbers
import time
from tqdm import tqdm

# Assuming that data is in same directory as code
DATA_DIRECTORY = os.getcwd()
OutCats = 4 # Magic number = dimension of encoded features

def encode(data, col: str) -> None:
    """
Takes a categorical column data from a dataframe and reduces the dimensionality
"""
    if "final_result" in data.columns:
        # We can create a simple vector to encode the data
        # We map the input column to strings to ensure a consistent type for np.unique
        oldDims, newDims = np.unique(data[col].transform(lambda k: str(k))), np.unique(data["final_result"])
        newDimIndex = {dim:i for i, dim in enumerate(newDims)}
        
        mapping = {o:[0 for _ in newDims] for o in oldDims}
        
        # Sum results to create the new vectors
        for _, row in data.iterrows():
            # Cast row[col] to string in case nan. Temporary solution and might be better to use a defaultDict
            mapping[str(row[col])][newDimIndex[row["final_result"]]] += 1

        # Normalise data
        print("Normalising")
        for cat, vector in mapping.items():
            total = sum(vector)
            for i in range(len(vector)):
                vector[i] /= total
        print(mapping)

        newSeries = [[] for _ in range(len(newDims))]
        for _, cat in data[col].items():
            vector = mapping[str(cat)]
            for v, s in zip(vector, newSeries):
                s.append(v)

        for i, s in enumerate(newSeries):
            data.loc[:, "{}_{}".format(col, i)] = pd.Series(s, index=data.index)
        return
    else:
        print("Not yet implemented...")
        return

def str_to_num(data, col: str) -> None:
    """
Maps categorical data to a number to allow it's use in a Random Forest Classifier
"""
    if not col in data.columns:
        print("Error: {} does not include {}".format(data.columns, col))
    # Empty dictionary will be built dynamically to contain the full mapping
    mapping = {}

    # New column to ultimately be added to data
    newSeries = []

    # Iteratively build up mapping and newSeries together
    counter = 0
    for x in data[col]:
        # Only need to convert if nan
        if (not isinstance(x, numbers.Number)) or np.isnan(x):
            # Always consider string representation of x for consistency
            key = str(x)
            # key might not exist in mapping yet
            try:
                y = mapping[key]
            except KeyError:
                mapping[key] = counter
                counter += 1
                y = mapping[key]
            finally:
                newSeries.append(y)
        else:
            newSeries.append(x)

    # Add new fully numeric series to the original DataFrame
    data.loc[:, "{}_numbers".format(col)] = pd.Series(newSeries, index=data.index)

    return

########################################################################
# SECTION ONE                                                          #
# ===========                                                          #
#                                                                      #
# In this section we load the csv files in to memory so they are       #
# available for processing in the rest of the program.                 #
#                                                                      #
# We expect to load in the following files:                            #
#   -assessments.csv                                                   #
#   -courses.csv                                                       #
#   -studentAssessment.csv                                             #
#   -studentInfo.csv                                                   #
#   -studentRegistration.csv                                           #
#   -studentVle.csv                                                    #
#   -vle.csv                                                           #
#                                                                      #
########################################################################

# This dictionary will contain every DataFrame object
data_dict = {}
start = time.time()

# Load every .csv file into memory
print("Loading all .csv files")
for filename in os.listdir(DATA_DIRECTORY):
    if filename.endswith(".csv"):
        data_dict[filename[:-4]] = pd.read_csv(filename)

print("Time taken now up to {} seconds".format(time.time() - start))

# Print out head of each table to test that it has worked
for title, table in data_dict.items():
    print("    -{}".format(title))

########################################################################
# SECTION TWO                                                          #
# ===========                                                          #
#                                                                      #
# In this section we create several dictionary structures that will    #
# allow us to easily collect data such as average assessment marks and #
# behaviour in VLEs.                                                   #
#                                                                      #
# The dictionaries we create are:                                      #
#   -student_assess_count (id_student, [assessments completed...])     #
#   -student_vle_count (id_student, [vle behaviour recorded...])       #
#   -tma, cma (id_assessment, weighting)                               #
#   -assessment_marks (id_assessment, [marks achieved on this...])     #
#                                                                      #
########################################################################

# This dictionary will record the assessments completed by each student
print("Initialising student dictionaries")
student_assess_count = {}
student_vle_count = {}

# Should be more efficient than doing two dictionary comprehensions
for student in tqdm(data_dict["studentInfo"]["id_student"]):
    student_assess_count[student] = []
    student_vle_count[student] = []


# These dictionaries will contain the information needed to properly weight and record info about each assessment
print("Loading assessment data")
tma = {}
cma = {}
exams = []

assessment_marks = {}

for _, row in tqdm(data_dict["assessments"].iterrows()): #(index, Series)
    id_assess = row["id_assessment"]
    assess_type = row["assessment_type"]
    weight = row["weight"]
    
    if assess_type == "TMA":
        tma[id_assess] = weight
        cma[id_assess] = 0
    elif assess_type == "CMA":
        tma[id_assess] = 0
        cma[id_assess] = weight
    else:
        # All exams are weigted as 100%
        exams.append(row["id_assessment"])

    assessment_marks[id_assess] = [] # Empty list to record student performances at this assessment


########################################################################
# SECTION THREE                                                        #
# =============                                                        #
#                                                                      #
# In this section we count and calculate the various metrics that will #
# be used to build our model. The dictionaries created above will be   #
# able to record the data already in the DataFrames and will then      #
# allow various statistics to be calculated. Finally we create a       #
# single DataFrame (students) of all the data for the model.           #
#                                                                      #
# The data we will be collecting are:                                  #
#   -avg_score       An unweighted average of a student's assessments  #
#   -assess_count    The number of assessments done                    #
#   -weighted_scores The weighted average of a student's assessments   #
#   -exam_score      Students performance on exams                     #
#                                                                      #
########################################################################

# Go through a for loop to look at all assessments on a per student basis
print("Counting assessments for each student")
# Adding extra variables allows tqdm to print out a pretty progress bar for us - part 1
assess_iterations = len(data_dict["studentAssessment"])
assess_iterator = data_dict["studentAssessment"].iterrows()
for _ in tqdm(range(assess_iterations)):
    _, row = next(assess_iterator)
    student_assess_count[row.pop("id_student")].append(row)
    assessment_marks[row["id_assessment"]].append(row["score"])

#print("Counting vle data for each student")
## Adding extra variables allows tqdm to print out a pretty progress bar for us - part 2
#vle_iterations = len(data_dict["studentVle"])
#vle_iterator = data_dict["studentVle"].iterrows()
#for _ in tqdm(range(vle_iterations)):
#    _, row = next(vle_iterator)
#    student_vle_count[row.pop("id_student")].append(row)

# Create a single dataframe of all relevant data to pass to algorithm
print("Adding registration info to dataframe")
students = data_dict["studentInfo"]
for column in data_dict["studentRegistration"].columns:
    if column != "id_student":
        students.loc[:, column] = pd.Series(data_dict["studentRegistration"][column], index=students.index)

        
# Add columns based on assessment performance
# Num. of assessments done
mean = lambda marks: sum(marks)/len(marks) if len(marks) > 0 else 0
assess_mean = {id_assess:mean(assessment_marks[id_assess]) for id_assess in assessment_marks}
var = lambda marks, mean: sum([(m - mean)**2 for m in marks])/len(marks) if len(marks) > 0 else 0
assess_var  = {id_assess:var(assessment_marks[id_assess], assess_mean[id_assess]) for id_assess in assessment_marks}
        
# Average score on assessments
print("Averaging assessment scores for each student")
# This is not done with a list comprehension as that is unwieldly in terms of checking that valid inputs are used
assessment_count = []
average_scores = []
weighted_scores = []
exam_score = []
avg_var = []


for stu in tqdm(students["id_student"]):
    total = 0
    weighted_total = 0
    num = 0
    exam_count = 0
    exam_total = 0
    variance = 0
    for mark in student_assess_count[stu]:
        score = mark["score"]
        if not np.isnan(score):
            total += score
            
            id_assess = mark["id_assessment"]

            # Record assessment mark
            assessment_marks[id_assess].append(score)

            # Gather information to create an average of this student's performance
            if id_assess in exams:
                exam_total += score
                exam_count += 1
            else:
                weighted_total += score * (tma[id_assess] + cma[id_assess])/100 # Here the score is weighted
                
            num += 1

            # Check how this student averages against other students who did this assessment too
            variance += (score - assess_mean[id_assess])/assess_var[id_assess]
            
    # To prevent division by zero
    if num == 0:
        average_scores.append(0)
        weighted_scores.append(0)
        avg_var.append(0)
    else:
        average_scores.append(total/num)
        weighted_scores.append(weighted_total/num)
        avg_var.append(variance/num)

    if exam_count == 0:
        exam_score.append(0)
    else:
        exam_score.append(exam_total/exam_count)
    assessment_count.append(num)

print("Time taken now up to {time.time() - start} seconds")

students.loc[:, "assess_count"] = pd.Series(assessment_count, index=students.index)
students.loc[:, "avg_score"] = pd.Series(average_scores, index=students.index)
students.loc[:, "weighted_scores"] = pd.Series(weighted_scores, index=students.index)
students.loc[:, "exam_score"] = pd.Series(exam_score, index=students.index)
students.loc[:, "avg_var"] = pd.Series(avg_var, index=students.index)

print("Time taken now up to {time.time() - start} seconds")

########################################################################
# SECTION FOUR                                                         #
# ============                                                         #
#                                                                      #
# We can now select which features to pass to the model, encoding      #
# non-numeric data appropriately.                                      #
#                                                                      #
########################################################################

# Split dataset into training data and test data - this is a standard way of splitting data as per https://www.datacamp.com/community/tutorials/random-forests-classifier-python#building
print("Splitting data")
# Select default values
features = ["avg_score", "assess_count", "weighted_scores", "exam_score"]

# Add extra values that have been encoded
encFeats = []#"region", "imd_band", "highest_education", "gender"]
ntsFeats = ["region", "imd_band", "highest_education", "gender"]
for feat in encFeats:
    print("Encoding {}".format(feat))
    encode(students, feat)
    for i in range(OutCats):
        features.append("{}_{}".format(feat, i))
        
for s in ntsFeats:
    str_to_num(students, s)
    features.append("{}_numbers".format(s))

# This is the final DataFrame from which we create a training and testing set    
X = students[features]

print("Time taken now up to {time.time() - start} seconds")

# Library expects numerical labels so convert them using this mapping
results_mapper = {"Withdrawn":0, "Fail":1, "Pass":2, "Distinction":3}
y = students["final_result"].map(lambda k:results_mapper[k])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest
print("Preparing training")
rf_scores = []
rf_sizes = range(10, 11)
for n in rf_sizes:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    rf_scores.append(rf.score(X_test, y_test))


# Analyse data
from matplotlib import pyplot as plt
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xticks(range(len(features)), features, rotation = 25)


plt.savefig("features")
#plt.show()
