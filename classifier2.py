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

def howto():
    """This function will print an explanation of how the program is broken
    into various chunks, allowing for faster running within the IDLE environment
    """
    print('Run the howto() function to view this again.\n')
    print('Program currently runs with 6 function calls:')
    print('\t1) loadData() loads the csv files from the location specified in DATA_DIRECTORY and stores them in the dictionary called data;')
    print('\t2) gatherAssInfo() creates a dictionary assInfo containing information about each assessment in assessments.csv;')
    print('\t3) countStuMarks() builds two dictionaries storing assessment performances for each student on coursework and exams, respectively;')
    print('\t4) enrollmentCount() builds the enrollment module showing how many students are enrolled in each module;')
    print('\t5) assNorms() calculates mean and variance data for each assessment so that student score can be normalised;')
    print('\t6) coallesce() uses the data from previous steps to create a new dataframe, dataset, where categorical features are converted to numerical ones.')

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
def loadData():         # Step 1
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
#       each region can be expressed as 4 features showing how often    #
#       it receives each result compared to the national average        #
#   -highest_education:                                                 #
#       can be encoded with three bits                                  #
#   -imd_band:                                                          #
#       each option is a range so simply use median value for each      #
#       we set missing values to 50                                     #
#   -disability: Y = 1, N = 0                                           #
#                                                                       #
#########################################################################

# Firstly compile data from studentAssessment.csv calculating weighted coursework scores and exam scores

# Need to build a dictionary of assessment info
assInfo = {}
expectedAss = defaultdict(list)     # (module, presentation): [assessment_ids...]
def gatherAssInfo():    # Step 2
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
def countStuMarks():    # Step 3
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
def enrollmentCount():  # Step 4
    enrollBar = tqdm(iterable=data['studentInfo'].iterrows(), total=len(data['studentAssessment']))
    enrollBar.set_description('Counting enrollment per course')
    for _, row in enrollBar:
        enrollment[(row['code_module'], row['code_presentation'])] += 1
    del enrollBar

def assNorms():         # Step 5
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


def coallesce():        # Step 6
    year, month, modBit0, modBit1, modBit2, edu_bit0, edu_bit1, edu_bit2 = [], [], [], [], [], [], [], []

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
        edu_bit0.append(eb0Val)
        edu_bit1.append(eb1Val)
        edu_bit2.append(eb2Val)

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

    # When new columns were generated they must be added to the dataframe
    dataset['year'] = year
    dataset['month'] = month

    dataset['modBit0'] = modBit0
    dataset['modBit1'] = modBit1
    dataset['modBit2'] = modBit2

    dataset['edu_bit0'] = edu_bit0
    dataset['edu_bit1'] = edu_bit1
    dataset['edu_bit2'] = edu_bit2

    del year, yearVal
    del month, monthVal

    del modBit0, mb0Val
    del modBit1, mb1Val
    del modBit2, mb2Val

    del edu_bit0, eb0Val
    del edu_bit1, eb1Val
    del edu_bit2, eb2Val

    del stuInfobar

# Run the steps defined above
howto()
loadData()
gatherAssInfo()
countStuMarks()
enrollmentCount()
assNorms()

dataset = data['studentInfo'].copy()
dataset['courseworkScore'] = 0
dataset['examScore'] = 0
dataset['normCourseworkScore'] = 0
dataset['normExamScore'] = 0
coallesce()

try:
    dataset.to_csv('processed.csv')
except PermissionError:
    print('Failed to update processed.csv')
finally:
    print('Data converted')
    print(dataset['examScore'])
    print(dataset['courseworkScore'])