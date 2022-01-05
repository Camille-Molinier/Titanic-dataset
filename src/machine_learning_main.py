print("Modules importation :\n")

print(f"{'    Classical modules' :-<50}", end="")

from os import name
from re import A, X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import csv
import seaborn as sns
print(" Done\n")


print(f"{'    Sklearn modules' :-<50}", end="")
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report; plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures


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
# print(df.head(),'\n')

m_df = df[df['Sex'] == 'male']
f_df = df[df['Sex'] == 'female']

# print(df['Sex'].value_counts(),'\n')
# print(m_df.shape)
# print(f_df.shape, '\n')

## Train Test Split
m_trainset, m_testset = train_test_split(m_df, test_size=0.2, random_state=0)
f_trainset, f_testset = train_test_split(f_df, test_size=0.2, random_state=0)

########## Encoding ##########
dic_param = {
    'male' : 1, 'female' : 0,
    'S': 0, 'C': 1, 'Q': 2
}

def encode(_df):
    for col in _df.select_dtypes(object):
        _df[col] = _df[col].map(dic_param)
    
    return _df

def impute (_df):
    return _df.dropna()

def preprocessing (_df):
    _df = encode(_df)
    _df = impute(_df)

    X = _df.drop('Survived', axis=1)
    y = _df['Survived']

    return X, y

m_X_train, m_y_train = preprocessing(m_trainset)
m_X_test, m_y_test = preprocessing(m_testset) 

f_X_train, f_y_train = preprocessing(f_trainset)
f_X_test, f_y_test = preprocessing(f_testset) 

# print(m_X_train.head(), '\n')

# print('Male X train   :', m_X_train.shape, m_y_train.shape)
# print('Male X test    :', m_X_test.shape, m_y_test.shape, '\n')

# print('Female X train :', f_X_train.shape, f_y_train.shape)
# print('Female X test  :', f_X_test.shape, f_y_test.shape, '\n')


################################################################################
#                                  Modelization                                #
################################################################################
print("Modelization :")

def evaluation (model, name, X_train, y_train, X_test, y_test) :
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # print(classification_report(y_test, y_pred))

    # plt.figure()
    # plot_confusion_matrix(model, X_test, y_test)
    # plt.savefig(f'../dat/fig/models/{name} matrix.png')

    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))
    # plt.figure()
    # plt.plot(N, train_score.mean(axis=1), label='Train score')
    # plt.plot(N, val_score.mean(axis=1), label='Val score')
    # plt.legend()
    # plt.savefig(f'../dat/fig/models/{name} learning curve')

m_decision_tree = DecisionTreeClassifier(random_state=0)
f_decision_tree = DecisionTreeClassifier(random_state=0)

evaluation(m_decision_tree, 'Male Decision Tree', m_X_train, m_y_train, m_X_test, m_y_test)
evaluation(f_decision_tree, 'Female Decision Tree', f_X_train, f_y_train, f_X_test, f_y_test)




################################################################################
#                                   Submission                                 #
################################################################################
print("Submission :")

sub_data = pd.read_csv("../dat/test.csv")
sub_df = sub_data.drop(['Name', 'Ticket', 'Cabin', 'Age'], axis=1)

m_sub_df = sub_df[sub_df['Sex'] == 'male']
print(m_sub_df.shape)
# f_sub_df = sub_df[sub_df['Sex'] == 'female']

m_id = pd.DataFrame(m_sub_df['PassengerId'], columns=['PassengerId'])
# f_id = f_sub_df['PassengerId']

m_enc_sub_df = (encode(m_sub_df)).fillna(m_sub_df.mean()).drop('PassengerId', axis=1)
# f_enc_sub_df = (encode(f_sub_df)).fillna(f_sub_df.mean()).drop('PassengerId', axis=1)

m_y_pred = m_decision_tree.predict(m_enc_sub_df)
# f_y_pred = f_decision_tree.predict(f_enc_sub_df)

m_y_pred_df = pd.DataFrame(m_y_pred, columns=['Survived'])

m_sub_res = pd.concat([m_id,m_y_pred_df.reindex(m_id.index)], axis=1, ignore_index=True)

print(m_sub_res.head())

print(m_sub_res.shape)
m_sub_res.to_csv('./m_sub.csv', index=False)
