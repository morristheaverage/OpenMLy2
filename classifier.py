#########################################################################
#                                                                       #
#   ####     ####    ####                                               #
#   #####   #####    ####                                               #
#   #############    ####                                               #
#   #### ### ####    ####                                               #
#   ####     ####    ####                                               #
#   ####     ####    ##########                                         #
#   ####     ####    ##########                                         #
#                                                                       #
#########################################################################

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

def howto():
    """This function will print an explanation of how the program is broken
    into various chunks, allowing for faster running within the IDLE environment
    """
    print('Run the howto() function to view this again.\n')
    print('Program currently runs with the following function calls:')
    print('\t1) loadData() loads the csv files from the location specified in DATA_DIRECTORY and stores them in the dictionary called data;')
    print('\t2) gatherAssInfo() creates a dictionary assInfo containing information about each assessment in assessments.csv;')
    print('\t3) countStuMarks() builds two dictionaries storing assessment performances for each student on coursework and exams, respectively;')
    print('\t4) enrollmentCount() builds the enrollment module showing how many students are enrolled in each module;')
    print('\t5) assNorms() calculates mean and variance data for each assessment so that student score can be normalised;')
    print('\t6) countVle() counts totals of vle interactions per student')
    print('\t6) coallesce() uses the data from previous steps to create a new dataframe, dataset, where categorical features are converted to numerical ones.')
    print('\t7) datasetClean() drops the unnecessary columns from dataset')
    print('\t8) normalise() normalises non-binary features in dataset')
    print('\t9) Then various models and feature selections occur')

#########################################################################
# SECTION ONE                                                           #
# ===========                                                           #
#                                                                       #
# In this section we load the csv files in to memory so they are        #
# available for processing in the rest of the program.                  #
#                                                                       #
# We expect to load in the following files:                             #
#   -assessments.csv                                                    #
#   -courses.csv                                                        #
#   -studentAssessment.csv                                              #
#   -studentInfo.csv                                                    #
#   -studentRegistration.csv                                            #
#   -studentVle.csv                                                     #
#   -vle.csv                                                            #
#                                                                       #
#########################################################################

# Import data from local directory
DATA_DIRECTORY = os.getcwd()

# We will store each pandas dataframe in a dictionary
data = {}
def loadData():             # Step 1
    loadbar = tqdm(os.listdir(DATA_DIRECTORY))
    loadbar.set_description('Reading csv files')
    for filename in loadbar:
        # Load all csv files and save with suitable name
        if filename.endswith('.csv'):
            data[filename[:-4]] = pd.read_csv(filename)
    del loadbar

    for title in data:
        print("    -{}".format(title))

#########################################################################
# SECTION TWO                                                           #
# ===========                                                           #
#                                                                       #
# We convert various categorical features into numerical data:          #
#   -specific module has 22 categories e.g. AAA 2013B                   #
#       solution is 5 binary features - year (13/14), month(B/J) and    #
#       3 single bit features to encode the 7 module codes              #
#   -age_band: simply pick a reasonable number within the range         #
#       25, 45 or 65 years old                                          #
#   -region:                                                            #
#       each region can be expressed as 4 features representing a       #
#       unique binary code                                              #
#   -highest_education:                                                 #
#       can be encoded with three bits                                  #
#   -imd_band:                                                          #
#       each option is a range so simply use median value for each      #
#       we set missing values to 50                                     #
#   -disability: Y = 1, N = 0                                           #
#   -gender: M = 1, F = 0                                               #
#                                                                       #
#########################################################################

# Firstly compile data from studentAssessment.csv calculating weighted coursework scores and exam scores

# Need to build a dictionary of assessment info
assInfo = {}
expectedAss = defaultdict(list)     # (module, presentation): [assessment_ids...]
def gatherAssInfo():        # Step 2
    assBar = tqdm(iterable=data['assessments'].iterrows(), total=len(data['assessments']))
    assBar.set_description('Load assessment info')
    for _, row in assBar:
        assInfo[row['id_assessment']] = {
            'code_module': row['code_module'],
            'code_presentation': row['code_presentation'],
            'type': row['assessment_type'],
            'weight': row['weight'],
            'scores': [] # Empty list to help normalise scores
            }
        expectedAss[(row['code_module'], row['code_presentation'])].append(row['id_assessment'])
    del assBar

courseworkMarks = defaultdict(list)
examMarks = defaultdict(list)

# Now we can fill the dictionaries creating lists of assessments
# for each student. A student can be registered for as many as 5 courses
# so it is important to check which course the assessment is for
def countStuMarks():        # Step 3
    marksBar = tqdm(iterable=data['studentAssessment'].iterrows(), total=len(data['studentAssessment']))
    marksBar.set_description('Record marks for each student')
    for _, row in marksBar:
        ass = row['id_assessment']
        stu = row['id_student']
        module, presentation = assInfo[ass]['code_module'], assInfo[ass]['code_presentation']

        score, weight = row['score'], assInfo[ass]['weight']
        if np.isnan(score):
            score = 0
        if assInfo[ass]['type'] == 'Exam':
            examMarks[(stu, module, presentation)].append((score, weight, ass)) # Weight for exam should be 100%
        else:
            courseworkMarks[(stu, module, presentation)].append((score, weight, ass))
        assInfo[ass]['scores'].append(score)
    del marksBar

# We can caluclate the mean and variance on each assessment to help normalise scores
# but first we need enrollment rates in each course
enrollment = defaultdict(int)
def enrollmentCount():      # Step 4
    enrollBar = tqdm(iterable=data['studentInfo'].iterrows(), total=len(data['studentAssessment']))
    enrollBar.set_description('Counting enrollment per course')
    for _, row in enrollBar:
        enrollment[(row['code_module'], row['code_presentation'])] += 1
    del enrollBar

def assNorms():             # Step 5
    normDataAssBar = tqdm(assInfo)
    normDataAssBar.set_description('Calculate mean and variance of each assessment')
    for ass in normDataAssBar:
        assData = assInfo[ass]
        scores = assData['scores']
        mod, pres = assData['code_module'], assData['code_presentation']
        tookAss = len(scores)
        if tookAss == 0:
            assInfo[ass]['mean'], assInfo[ass]['var'] = 0, 0
        else:
            # Calculate mean
            mean = sum(scores)/tookAss
            # Calculate variance
            var = sum((score - mean)**2 for score in scores)/tookAss
            assInfo[ass]['mean'], assInfo[ass]['var'] = mean, var
    del normDataAssBar

vleCounts = defaultdict(int)
def countVle():             # Step 6
    vleBar = tqdm(iterable=data['studentVle'].iterrows(), total=len(data['studentVle']))
    vleBar.set_description('Counting vle interactions')
    for _, row in vleBar:
        vleCounts[row['id_student']] += row['sum_click']

# Secondly add new features to data['studentInfo'] by converting existing
# categorical data in the table to numeric data and adding data from other tables
def weightedAvg(marks: list) -> float:  # Helper function for Step 6
    """Given a list of (score, weight, ...) tuples
    return the weighted average of all assessments
    """
    totalWeight = sum(x[1] for x in marks)
    if len(marks) == 0 or totalWeight == 0:
        return 0.0

    return sum(x[0]*x[1] for x in marks)/totalWeight


def coallesce():            # Step 7
    year, month = [], []
    modBit0, modBit1, modBit2 = [], [], []
    eduBit0, eduBit1, eduBit2 = [], [], []
    regBit0, regBit1, regBit2, regBit3 = [], [], [], []

    stuInfobar = tqdm(iterable=dataset.iterrows(), total=len(dataset))
    stuInfobar.set_description('Converting module feature')
    for i, row in stuInfobar:
        # Add features calculated from other tables
        stuID = row['id_student']
        module = row['code_module']
        presentation = row['code_presentation']

        courseworkTuples = courseworkMarks[(stuID, module, presentation)]
        examTuples = examMarks[(stuID, module, presentation)]

        expected = expectedAss[(module, presentation)]
        for ass in expected:
            # If any expected assignment is not recorded...
            if assInfo[ass]['type'] == 'Exam':
                if not any(True if tup[2] == ass else False for tup in examTuples):
                    examTuples.append((0, 100, ass))
            else:
                if not any(True if tup[2] == ass else False for tup in courseworkTuples):
                    # Record mark as zero
                    courseworkTuples.append((0, assInfo[ass]['weight'], ass))


        # mu = assInfo[tup[2]]['mean']
        # sigma = np.sqrt(assInfo[tup[2]]['var'])
        # score = tup[0]
        # weight = tup[1]
        # normScore = (score - mu) / sigma
        normCourseworkTuples = []
        for tup in courseworkTuples:
            mean, var = assInfo[tup[2]]['mean'], assInfo[tup[2]]['var']
            if var <= 0:
                normCourseworkTuples.append((0, 0))
            else:
                normCourseworkTuples.append((
                    (tup[0] - mean) / np.sqrt(var),
                    tup[1]
                    ))

        normExamTuples = []
        for tup in examTuples:
            mean, var = assInfo[tup[2]]['mean'], assInfo[tup[2]]['var']
            if var == 0:
                normExamTuples.append((0, 0))
            else:
                normExamTuples.append((
                    (tup[0] - mean) / np.sqrt(var),
                    tup[1]
                ))

        dataset.loc[i, 'courseworkScore'] = weightedAvg(courseworkTuples)
        dataset.loc[i, 'examScore'] = weightedAvg(examTuples)

        nca = weightedAvg(normCourseworkTuples)
        if np.isnan(nca):
            print('\n', normCourseworkTuples)
            print('vs\n', courseworkTuples)
            
        dataset.loc[i, 'normCourseworkScore'] = nca
        dataset.loc[i, 'normExamScore'] = weightedAvg(normExamTuples)

        # Vle counts
        dataset.loc[i, 'vleCount'] = vleCounts[stuID]

        # Some features need to be expressed in new columns
        yearVal = 1 if row['code_presentation'][3] == '4' else 0
        monthVal = 1 if row['code_presentation'][4] == 'J' else 0
        year.append(yearVal)
        month.append(monthVal)
        
        module_code = row['code_module'][0]
        mb0Val = 1 if module_code in ['E', 'F', 'G'] else 0
        mb1Val = 1 if module_code in ['C', 'D', 'G'] else 0
        mb2Val = 1 if module_code in ['B', 'D', 'F'] else 0
        modBit0.append(mb0Val)
        modBit1.append(mb1Val)
        modBit2.append(mb2Val)
        
        education = row['highest_education']
        eb0Val = 1 if education in ['Lower Than A Level', 'HE Qualification'] else 0
        eb1Val = 1 if education in ['A Level or Equivalent', 'HE Qualification'] else 0
        eb2Val = 1 if education in ['Post Graduate Qualification'] else 0
        eduBit0.append(eb0Val)
        eduBit1.append(eb1Val)
        eduBit2.append(eb2Val)

        region = row['region']
        rb0Val = 1 if region in ['South Region', 'South West Region', 'Wales', 'West Midlands Region', 'Yorkshire Region'] else 0
        rb1Val = 1 if region in ['North Region', 'North Western Region', 'Scotland', 'South East Region', 'Yorkshire Region'] else 0
        rb2Val = 1 if region in ['Ireland', 'London Region', 'Scotland', 'South East Region', 'Wales', 'West Midlands Region'] else 0
        rb3Val = 1 if region in ['East Midlands Region', 'London Region', 'North Western Region', 'South East Region', 'South West Region', 'West Midlands Region'] else 0
        regBit0.append(rb0Val)
        regBit1.append(rb1Val)
        regBit2.append(rb2Val)
        regBit3.append(rb3Val)

        # Other times exising columns can be altered
        # Age
        age_band = dataset.loc[i, 'age_band']
        if age_band == '0-35':
            age_num = 25
        elif age_band == '35-55':
            age_num = 45
        else:
            age_num = 65
        dataset.loc[i, 'age_band'] = age_num

        # IMD Band
        imd_band = dataset.loc[i, 'imd_band']
        imd_num = int(imd_band[0]) * 10 + 5 if type(imd_band) == str else 50
        dataset.loc[i, 'imd_band'] = imd_num

        # Disability
        dataset.loc[i, 'disability'] = 1 if row['disability'] == 'Y' else 0

        # Gender
        dataset.loc[i, 'gender'] = 1 if row['gender'] == 'M' else 0

    # When new columns were generated they must be added to the dataframe
    dataset['year'] = year
    dataset['month'] = month

    dataset['modBit0'] = modBit0
    dataset['modBit1'] = modBit1
    dataset['modBit2'] = modBit2

    dataset['eduBit0'] = eduBit0
    dataset['eduBit1'] = eduBit1
    dataset['eduBit2'] = eduBit2
    
    dataset['regBit0'] = regBit0
    dataset['regBit1'] = regBit1
    dataset['regBit2'] = regBit2
    dataset['regBit3'] = regBit3

    del year, yearVal
    del month, monthVal

    del modBit0, mb0Val
    del modBit1, mb1Val
    del modBit2, mb2Val

    del eduBit0, eb0Val
    del eduBit1, eb1Val
    del eduBit2, eb2Val

    del stuInfobar

def datasetClean():         # Step 8
    global dataset
    dataset = dataset.drop([
        'code_module',
        'code_presentation',
        'id_student',
        'region',
        'highest_education'
    ], axis=1)

def normalise(cols: list):  # Step 9
    global dataset
    # First calculate the total values for each row
    # to get mean and std deviation values
    totals = defaultdict(list)
    mvBar = tqdm(iterable=dataset.iterrows(), total=len(dataset))
    mvBar.set_description('Building stats')
    for _, row in mvBar:
        for col in cols:
            totals[col].append(row[col])
    means = {col: sum(totals[col])/len(totals[col]) if len(col) > 0 else 0 for col in totals}
    variances = {col: sum((val - means[col])**2 for val in totals[col])/len(totals[col]) if len(totals[col]) > 0 else 0 for col in totals}
    # Now apply stats to each row of dataset
    normieBar = tqdm(iterable=dataset.iterrows(), total=len(dataset))
    normieBar.set_description('Normalising dataset')
    for i, row in normieBar:
        for col in cols:
            current = row[col]
            dataset.loc[i, col] = (current - means[col])/np.sqrt(variances[col]) if variances[col] > 0 else 0

#########################################################################
# SECTION THREE                                                         #
# =============                                                         #
#                                                                       #
# We now have a training set and testing set of data. This includes     #
# features which are either normalised (mean: 0, variance: 1) or binary #
#                                                                       #
#           Normalised              |             Binary                #
# ==================================|================================== #
#            imd_band               |             gender                #
#            age_band               |           disability              #
#      num_of_prev_attempts         |              year                 #
#         studied_credits           |             month                 #
#         courseworkScore           |           modBit[0-2]             #
#            examScore              |           eduBit[0-2]             #
#       normCourseworkScore         |           regBit[0-3]             #
#          normExamScore            |                                   #
#            vleCount               |                                   #
#                                   |                                   #
#                                                                       #
#########################################################################

# Run the steps defined above
howto()
loadData()

# Assessments
gatherAssInfo()
countStuMarks()
enrollmentCount()
assNorms()

# Vles
countVle()

dataset = data['studentInfo'].copy()
dataset['courseworkScore'] = 0
dataset['examScore'] = 0
dataset['normCourseworkScore'] = 0
dataset['normExamScore'] = 0
dataset['vleCount'] = 0
coallesce()
datasetClean()
normalise([
    'imd_band',
    'age_band',
    'num_of_prev_attempts',
    'studied_credits',
    'courseworkScore',
    'examScore',
    'normCourseworkScore',
    'normExamScore',
    'vleCount'
])

try:
    dataset.to_csv('processed.csv')
except PermissionError:
    print('Failed to update processed.csv')
finally:
    print('Data converted')

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['final_result'], axis=1), dataset['final_result'],
    test_size=0.3, random_state=42
)
# We can turn y_train and y_test into three binary series each
wfMap = lambda res: 0 if res in ['Withdrawn', 'Fail'] else 1
wpMap = lambda res: 0 if res in ['Withdrawn', 'Pass'] else 1
wdMap = lambda res: 0 if res in ['Withdrawn', 'Distinction'] else 1

yWF_train, yWF_test = pd.Series(map(wfMap, y_train)), pd.Series(map(wfMap, y_test))
yWP_train, yWP_test = pd.Series(map(wpMap, y_train)), pd.Series(map(wpMap, y_test))
yWD_train, yWD_test = pd.Series(map(wdMap, y_train)), pd.Series(map(wdMap, y_test))

corrData = X_train.copy().drop([
    'num_of_prev_attempts', 'modBit0', 'modBit1', 'modBit2',
    'eduBit0', 'eduBit1', 'eduBit2', 'regBit0', 'regBit1', 'regBit2', 'regBit3',
    'year', 'month'
], axis=1)
corrData['y'] = yWF_train
correlation = corrData.corr()

from matplotlib import pyplot as plt
import seaborn as sns
# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
ax = sns.heatmap(
    correlation,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.title('Correlation heatmap')
plt.savefig('Heatmap')

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfcScore = rfc.score(X_test, y_test)

# Use feature importances to inform new model
importances = sorted([(importance, feature) for importance, feature in zip(rfc.feature_importances_, X_train.columns)], key=lambda k: k[0], reverse=True)
rfc_models = []
index = [i for i in range(len(importances))]
for i in index:
    toDrop = [tup[1] for tup in importances[i+1:]]
    print(toDrop)
    rfc.fit(X_train.drop(toDrop, axis=1), y_train)
    rfc_models.append(rfc.score(X_test.drop(toDrop, axis=1), y_test))

# Analyse data
plt.barh(range(len(importances)), [tup[0] for tup in importances])
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.yticks(range(len(importances)), [tup[1] for tup in importances])
plt.savefig("rfc_features")
#plt.show()

plt.scatter(rfc_models, index)
plt.title('Model performances with n most important features')
plt.ylabel('N features')
plt.xlabel('Test score')
for i, txt in enumerate(importances):
    plt.annotate(txt[1], (rfc_models[i] + 0.005, index[i]))

noise_train, noise_test = X_train.copy(), X_test.copy()
noise_train['noise'], noise_test['noise'] = np.random.normal(size=len(X_train)), np.random.normal(size=len(X_test))
noise_train['small_noise'], noise_test['small_noise'] = np.random.binomial(1, 0.995, size=len(X_train)), np.random.binomial(1, 0.995, size=len(X_test))
noise_train['large_noise'], noise_test['large_noise'] = np.random.binomial(1, 0.5, size=len(X_train)), np.random.binomial(1, 0.5, size=len(X_test))
noise_rfc = RandomForestClassifier(n_estimators=100, random_state=42)
noise_rfc.fit(noise_train, y_train)
noise_rfc.score(noise_test, y_test)

plt.barh(range(len(noise_rfc.feature_importances_)), noise_rfc.feature_importances_)
plt.title("Feature Importance with Noise")
plt.xlabel("Importance")
plt.yticks(range(len(noise_train.columns)), noise_train.columns)

# Refined rfc model
ref_rfc = RandomForestClassifier(n_estimators=100, random_state=42)
ref_rfc.fit(X_train.filter(['normCourseworkScore', 'normExamScore'], axis=1), y_train)
print('With 2 features score = ', ref_rfc.score(X_test.filter(['normCourseworkScore', 'normExamScore'], axis=1), y_test))
ref_rfc.fit(X_train.filter(['normCourseworkScore', 'normExamScore', 'courseworkScore', 'examScore'], axis=1), y_train)
print('With 4 features score = ', ref_rfc.score(X_test.filter(['normCourseworkScore', 'normExamScore', 'courseworkScore', 'examScore'], axis=1), y_test))
ref_rfc.fit(X_train.filter(['normCourseworkScore', 'normExamScore', 'vleCount'], axis=1), y_train)
print('With 3 features score = ', ref_rfc.score(X_test.filter(['normCourseworkScore', 'normExamScore', 'vleCount'], axis=1), y_test))



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
logr = LogisticRegression(random_state=42, solver='liblinear')
# We want to pick the n best features based on various criteria
# First criteria feature importance from random forest
threshold = 0.05
toDrop = [col for col, importance in zip(X_train.columns, rfc.feature_importances_) if importance < threshold]
logr.fit(X_train.drop(toDrop, axis=1), y_train)
print('Score of logistic regression when using features of importance above threshold: {} as rated by  random forest: '.format(threshold), logr.score(X_test.drop(toDrop, axis=1), y_test))


balanced_logr = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
feats = ['normCourseworkScore', 'normExamScore', 'vleCount', 'imd_band', 'studied_credits']
logr.fit(X_train.filter(feats, axis=1), yWF_train)
balanced_logr.fit(X_train.filter(feats, axis=1), yWF_train)
print('Score: ', logr.score(X_test.filter(feats, axis=1), yWF_test))
print('Score: ', balanced_logr.score(X_test.filter(feats, axis=1), yWF_test))


# Hyperparameters are fun
trees = [t for t in range(1, 102, 5)]
for tree_count in trees:
    forest = RandomForestClassifier(n_estimators=tree_count, random_state=42)
    forest.fit(X_train.filter(feats, axis=1), y_train)
    print('With {} trees: '.format(tree_count), forest.score(X_test.filter(feats, axis=1), y_test))

from sklearn.metrics import confusion_matrix
pred = balanced_logr.predict(X_train.filter(feats, axis=1))
confused = confusion_matrix(yWF_train, pred)
print('Confusion Matrix ', confused)
