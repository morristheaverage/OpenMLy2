#########################################################################
#                                                                       #
#   ####     ####    ####                                               #
#   #####   #####    ####                                               #
#   #############    ####                                               #
#   #### ### ####    ####                                               #
#   ####     ####    ####                                               #
#   ####     ####    #########                                          #
#   ####     ####    #########                                          #
#                                                                       #
#########################################################################

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict

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
loadbar = tqdm(os.listdir(DATA_DIRECTORY))
loadbar.set_description('Reading csv files')
for filename in loadbar:
    # Load all csv files and save with suitable name
    if filename.endswith('.csv'):
        data[filename[:-4]] = pd.read_csv(filename)
del loadbar

for title, table in data.items():
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

# First add new features to data['studentInfo'] by converting existing
# categorical data in the table to numeric data
year, month, modBit0, modBit1, modBit2, edu_bit0, edu_bit1, edu_bit2 = [], [], [], [], [], [], [], []

stuInfobar = tqdm(iterable=data['studentInfo'].iterrows(), total=len(data['studentInfo']))
stuInfobar.set_description('Converting module feature')
for i, row in stuInfobar:
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
    age_band = data['studentInfo'].loc[i, 'age_band']
    if age_band == '0-35':
        age_num = 25
    elif age_band == '35-55':
        age_num = 45
    else:
        age_num = 65
    data['studentInfo'].loc[i, 'age_band'] = age_num

    # IMD Band
    imd_band = data['studentInfo'].loc[i, 'imd_band']
    imd_num = int(imd_band[0]) * 10 + 5 if type(imd_band) == str else 50
    data['studentInfo'].loc[i, 'imd_band'] = imd_num

    # Disability
    data['studentInfo'].loc[i, 'disability'] = 1 if row['disability'] == 'Y' else 0

# When new columns were generated they must be added to the dataframe
data['studentInfo']['year'] = year
data['studentInfo']['month'] = month

data['studentInfo']['modBit0'] = modBit0
data['studentInfo']['modBit1'] = modBit1
data['studentInfo']['modBit2'] = modBit2

data['studentInfo']['edu_bit0'] = edu_bit0
data['studentInfo']['edu_bit1'] = edu_bit1
data['studentInfo']['edu_bit2'] = edu_bit2

del year, yearVal
del month, monthVal

del modBit0, mb0Val
del modBit1, mb1Val
del modBit2, mb2Val

del edu_bit0, eb0Val
del edu_bit1, eb1Val
del edu_bit2, eb2Val

del stuInfobar

print('Data converted')
print(data['studentInfo'].columns)


# Secondly add data from studentAssessment.csv calculating weighted coursework scores and exam scores
data['studentInfo']['courseworkScore'] = np.nan
data['studentInfo']['examScore'] = np.nan

# Need to build a dictionary of assessment info
assInfo = {}
assBar = tqdm(iterable=data['assessments'].iterrows(), total=len(data['assessments']))
assBar.set_description('Load assessment info')
for _, row in assBar:
    assInfo[row['id_assessment']] = {
        'type': row['assessment_type'],
        'weight': row['weight']
        }
del assBar

courseworkMarks = defaultdict(list)
examMarks = defaultdict(list)

# Now we can fill the dictionaries creating lists of assessments
# for each student
marksBar = tqdm(iterable=data['studentAssessment'].iterrows(), total=len(data['studentAssessment']))
marksBar.set_description('Record marks for each student')
for _, row in marksBar:
    ass = row['id_assessment']
    stu = row['id_student']
    if assInfo[ass]['type'] == 'Exam':
        examMarks[stu].append(row['score'])
    else:
        courseworkMarks[stu].append((row['score'], assInfo[ass]['weight']))
del marksBar