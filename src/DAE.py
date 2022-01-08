import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("Data Loading :")

df = pd.read_csv("../dat/train.csv")
print("Successfully loading data !\n\n") #, data.head())
pd.set_option('display.max_columns', 100)
plt.rcParams.update({'figure.max_open_warning' : 30})

########## Formal exploration ##########
# Number of fows and columns
print("\nData shape : ", df.shape)

# Variables types
print("\nVariables types :\n", df.dtypes)

# Number of each types
print("\nDtypes value counts :\n", df.dtypes.value_counts())

# Graphical representation
plt.figure()
df.dtypes.value_counts().plot.pie()
plt.title("Dtypes value counts")
plt.savefig("../dat/fig/DEA/1_dtypes_value_counts.png")

# Missing values
plt.figure()
sns.heatmap(df.isna(), cbar=False)
plt.title("Missing values")
plt.savefig("../dat/fig/DEA/1_missing_values.png")

print("Missing values :\n", (df.isna().sum()/df.shape[0]).sort_values())

# Cabin na > 0.7 => Useless feature
df = df.drop('Cabin', axis=1)

########### Deep exploration ###########
## Target visualisation 'Survived'
print(df['Survived'].value_counts())

# Features Visualisation
# Continious features histogram
for col in df.select_dtypes(np.int64, np.float64):
    plt.figure()
    sns.distplot(df[col])
    plt.title(f"{col} histogram")
    plt.savefig(f"../dat/fig/DEA/2_{col}_histogram.png")
# Object features
dfo = df.drop(['Name', 'Ticket'], axis=1)
for col in dfo.select_dtypes(object):
    print(f'{col :-<50}{dfo[col].unique()}')

## Other grpahs
#Heatmap
plt.figure()
sns.heatmap(df.corr())
plt.savefig(f"../dat/fig/DEA/3_Heatmap.png")

# # Pairplot
plt.figure()
sns.pairplot(df, hue='Survived', diag_kws={'bw': 0.2})
plt.savefig(f"../dat/fig/DEA/3_Pairplot_Survived.png")

# plt.figure()
sns.pairplot(df, hue='Sex', diag_kws={'bw': 0.2})
plt.savefig(f"../dat/fig/DEA/3_Pairplot_Sex.png")
