import time
start = time.time()

import os
os.system('cls')

print("Modules importation :\n")
print(f"{'    Standard modules' :-<50}", end="")
from os import name, stat
from re import A, X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import csv
import seaborn as sns
from preprocessor import preprocessor
print(" Done\n")

print(f"{'    Sklearn modules' :-<50}", end="")
from sklearn.tree import DecisionTreeClassifier
from model_trainer import model_trainer
print(" Done\n")


################################################################################
#                                 Data Loading                                 #
################################################################################
print("Data Loading :")

data = pd.read_csv("../dat/train.csv")
print("Successfully loading data !\n\n") #, data.head())
pd.set_option('display.max_columns', 100)
plt.rcParams.update({'figure.max_open_warning' : 30})


################################################################################
#                           Data exploratory analysis                          #
################################################################################
# print("\nData exploratory analysis :")

# dfe = data.copy()

########## Formal exploration ##########
# Number of fows and columns
# print("\nData shape : ", dfe.shape)

# Variables types
# print("\nVariables types :\n", dfe.dtypes)

# Number of each types
# print("\nDtypes value counts :\n", dfe.dtypes.value_counts())

# Graphical representation
# plt.figure()
# dfe.dtypes.value_counts().plot.pie()
# plt.title("Dtypes value counts")
# plt.savefig("../dat/fig/DEA/1_dtypes_value_counts.png")

# Missing values
# plt.figure()
# sns.heatmap(dfe.isna(), cbar=False)
# plt.title("Missing values")
# plt.savefig("../dat/fig/DEA/1_missing_values.png")

# print("Missing values :\n", (dfe.isna().sum()/dfe.shape[0]).sort_values())

# Cabin na > 0.7 => Useless feature
# dfe = dfe.drop('Cabin', axis=1)

########### Deep exploration ###########
## Target visualisation 'Survived'
# print(dfe['Survived'].value_counts())

# Features Visualisation
# Continious features histogram
# for col in dfe.select_dtypes(np.int64, np.float64):
#     plt.figure()
#     sns.distplot(dfe[col])
#     plt.title(f"{col} histogram")
#     plt.savefig(f"../dat/fig/DEA/2_{col}_histogram.png")
# # Object features
# dfo = dfe.drop(['Name', 'Ticket'], axis=1)
# for col in dfo.select_dtypes(object):
#     print(f'{col :-<50}{dfo[col].unique()}')

## Other grpahs
#Heatmap
# plt.figure()
# sns.heatmap(dfe.corr())
# plt.savefig(f"../dat/fig/DEA/3_Heatmap.png")

# # Pairplot
# plt.figure()
# sns.pairplot(dfe, hue='Survived', diag_kws={'bw': 0.2})
# plt.savefig(f"../dat/fig/DEA/3_Pairplot_Survived.png")

# plt.figure()
# sns.pairplot(dfe, hue='Sex', diag_kws={'bw': 0.2})
# plt.savefig(f"../dat/fig/DEA/3_Pairplot_Sex.png")


################################################################################
#                                 Preprocessing                                #
################################################################################
print("Preprocessing :")

df = data.copy()
df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Age'], axis=1)

m_df = df[df['Sex'] == 'male']
f_df = df[df['Sex'] == 'female']

male_preprocessor = preprocessor(sex=True)
female_preprocessor = preprocessor(sex=False)

m_X_train, m_y_train, m_X_test, m_y_test = male_preprocessor.preprocessing(m_df)

f_X_train, f_y_train, f_X_test, f_y_test = female_preprocessor.preprocessing(f_df)

print(m_X_train.head(), '\n')

print('Male X train   :', m_X_train.shape, m_y_train.shape)
print('Male X test    :', m_X_test.shape, m_y_test.shape, '\n')

print('Female X train :', f_X_train.shape, f_y_train.shape)
print('Female X test  :', f_X_test.shape, f_y_test.shape, '\n')


################################################################################
#                                  Modelization                                #
################################################################################
print("Modelization :")

m_decision_tree = DecisionTreeClassifier(random_state=0)
f_decision_tree = DecisionTreeClassifier(random_state=0)

trainer = model_trainer()

trainer.evaluation(m_decision_tree, 'Male Decision Tree', m_X_train, m_y_train, m_X_test, m_y_test)
trainer.evaluation(f_decision_tree, 'Female Decision Tree', f_X_train, f_y_train, f_X_test, f_y_test)


################################################################################
#                                   Submission                                 #
################################################################################
# print("Submission :")

# # Loading data
# sub_data = pd.read_csv("../dat/test.csv")
# # Drop useless features
# sub_df = sub_data.drop(['Name', 'Ticket', 'Cabin', 'Age'], axis=1)

# # Split male and female
# m_sub_df = sub_df[sub_df['Sex'] == 'male']
# f_sub_df = sub_df[sub_df['Sex'] == 'female']

# # Grab the IDs
# m_id = pd.DataFrame(m_sub_df['PassengerId'], columns=['PassengerId'])
# m_sub_df2 = m_sub_df.drop(['PassengerId'], axis=1)
# f_id = pd.DataFrame(f_sub_df['PassengerId'], columns=['PassengerId'])
# f_sub_df2 = f_sub_df.drop(['PassengerId'], axis=1)

# # Encode datas
# m_enc_sub_df = (encode(m_sub_df2)).fillna(m_sub_df2.mean())
# f_enc_sub_df = (encode(f_sub_df2)).fillna(f_sub_df2.mean())

# # Make predictions
# m_y_pred = m_decision_tree.predict(m_enc_sub_df)
# f_y_pred = f_decision_tree.predict(f_enc_sub_df)

# # Concat IDs and predictions
# m_id = list(m_id['PassengerId'])
# f_id = list(f_id['PassengerId'])

# m_d = {'PassengerID' : m_id, 'Survived' : m_y_pred}
# f_d = {'PassengerID' : f_id, 'Survived' : f_y_pred}

# m_sub_res = pd.DataFrame(m_d)
# f_sub_res = pd.DataFrame(f_d)

# # Concat male and female results
# sub_res = pd.concat([m_sub_res, f_sub_res])

# # Export Results
# sub_res.to_csv('./submission.csv', index=False)


print(f'Processing complete (time : {round(time.time()-start, 4)}s)')
