import time
start = time.time()

import os
os.system('cls')

import warnings
warnings.filterwarnings('ignore')

print("Modules importation :\n")
print(f"{'    Standard modules' :-<50}", end="")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print(" Done\n")

print(f"{'    Sklearn modules' :-<50}", end="")
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
print(" Done\n")


################################################################################
#                                 Data Loading                                 #
################################################################################
print(f"{'Loading data ' :-<51}", end="")

# Read data from csv file
df = pd.read_csv("../dat/train.csv")

# Set options for displaying
pd.set_option('display.max_columns', 100)
plt.rcParams.update({'figure.max_open_warning' : 30})

print('Done\n')


################################################################################
#                                 Preprocessing                                #
################################################################################
print(f"{'Preprocessing ' :-<51}", end="")

# Create a copy of the data
df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Age'], axis=1)

# Spliting males and females
m_df = df[df['Sex'] == 'male']
f_df = df[df['Sex'] == 'female']

## Train Test Split
m_trainset, m_testset = train_test_split(m_df, test_size=0.2, random_state=0)
f_trainset, f_testset = train_test_split(f_df, test_size=0.2, random_state=0)

############# Encoding ############
dic_param = {
    'male' : 1, 'female' : 0,
    'S': 0, 'C': 1, 'Q': 2
}

def encode(_df):
    # For each column => Apply the dictionnary
    for col in _df.select_dtypes(object):
        _df[col] = _df[col].map(dic_param)

    return _df

############# Imputing ############
def impute (_df):
    # Drop all rows with at least one nan
    return _df.dropna()

########## Preprocessing ##########
def preprocessing (_df, sex):
    # If it's the male dataset
    if sex :
        _df = encode(_df)
        _df = impute(_df)
        
    # else it's the female dataset
    else :
        _df = encode(_df)
        _df = impute(_df)

    # Split features and target
    X = _df.drop('Survived', axis=1)
    y = _df['Survived']
    return X, y

###################################
# Apply preprocessing
m_X_train, m_y_train = preprocessing(m_trainset, True)
m_X_test, m_y_test = preprocessing(m_testset, True) 
f_X_train, f_y_train = preprocessing(f_trainset, False)
f_X_test, f_y_test = preprocessing(f_testset, False) 

print('Done\n')


################################################################################
#                                  Modelization                                #
################################################################################
# print(f"{'Modelization ' :-<51}", end="")
print('Modelization : \n')

def evaluation (model, name, X_train, y_train, X_test, y_test) :
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print the classification report
    report = classification_report(y_test, y_pred)
    print(report)
    with open(f'../dat/fig/models/{name}/classification_report.txt', "w") as file:
            file.write(report)
    

    # Create and save confusion matrix
    plt.figure()
    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig(f'../dat/fig/models/{name}/confusion_matrix.png')

    # Create and save learning curve
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))
    plt.figure()
    plt.plot(N, train_score.mean(axis=1), label='Train score')
    plt.plot(N, val_score.mean(axis=1), label='Val score')
    plt.legend()
    plt.savefig(f'../dat/fig/models/{name}/learning_curve')


########## Male model optimisation ##########
# Create a list of models
m_RandomForest = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
m_AdaBoost = make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=0))
# m_SVC = make_pipeline(StandardScaler(), SVC(random_state=0))
m_KNN = make_pipeline(StandardScaler(), KNeighborsClassifier())

# Creating gridSearch pipelines
hyper_params_RF = {'randomforestclassifier__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                   'randomforestclassifier__criterion': ['gini', 'entropy']}
hyper_params_Ada = {'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
                    'adaboostclassifier__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'adaboostclassifier__base_estimator': [DecisionTreeClassifier(random_state=0), KNeighborsClassifier()]}
hyper_params_KNN = {'kneighborsclassifier__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'kneighborsclassifier__leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'kneighborsclassifier__weights': ['uniform', 'distance'],
                    'kneighborsclassifier__algorithm': ['ball_tree', 'kd_tree', 'brute']}

m_grid_RF = GridSearchCV(m_RandomForest, hyper_params_RF, scoring='precision', cv=4)
m_grid_Ada = GridSearchCV(m_AdaBoost, hyper_params_Ada, scoring='precision', cv=4)
m_grid_KNN = GridSearchCV(m_KNN, hyper_params_KNN, scoring='precision', cv=4)


m_list_of_model = [[m_grid_RF, 'RandomForest'], [m_grid_Ada, 'AdaBoost'], [m_grid_KNN, 'KNN']] #,  [m_SVC, 'SVC']]

for model in m_list_of_model :
    print('  ', model[1], ' :')
    model[0].fit(m_X_train, m_y_train)
    best = model[0].best_params_
    with open(f'../dat/fig/models/{model[1]}/best_params.txt', "w") as file:
            file.write(best)
    evaluation(model[0].best_estimator_, f'Male/{model[1]}', m_X_train, m_y_train, m_X_test, m_y_test)

######### Female model optimisation #########

print('Done\n')


################################################################################
#                                   Submission                                 #
################################################################################
print(f"{'Submission ' :-<51}", end="")

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

# # Creating dataFrame [id, pred]
# m_d = {'PassengerID' : m_id, 'Survived' : m_y_pred}
# f_d = {'PassengerID' : f_id, 'Survived' : f_y_pred}
# m_sub_res = pd.DataFrame(m_d)
# f_sub_res = pd.DataFrame(f_d)

# # Concat male and female results
# sub_res = pd.concat([m_sub_res, f_sub_res])

# # Export Results
# sub_res.to_csv('./submission.csv', index=False)

print('Done\n')


print(f'Processing complete (time : {round(time.time()-start, 4)}s)')
