import pandas as pd
from sklearn.model_selection import train_test_split

class preprocessor :
################### Constructor ###################
    def __init__(self, sex):
        self.sex = sex
        self.dic_param = { 'male' : 1, 'female' : 0,
                           'S': 0, 'C': 1, 'Q': 2 }

############# Preprocessing functions #############
    def preprocessing(self, df) :
        trainset, testset = train_test_split(df, test_size=0.2, random_state=0)

        if self.sex :
            X_train, y_train = self.male_preprocessing(trainset)
            X_test, y_test = self.male_preprocessing(testset)
        else :
            X_train, y_train = self.female_preprocessing(trainset)
            X_test, y_test = self.female_preprocessing(testset)

        return X_train, y_train, X_test, y_test
    

    def male_preprocessing(self, df):
        df = self.male_encode(df)
        df = self.male_impute(df)

        X = df.drop('Survived', axis=1)
        y = df['Survived']

        return X, y

    def female_preprocessing(self, df) :
        df = self.female_encode(df)
        df = self.female_impute(df)

        X = df.drop('Survived', axis=1)
        y = df['Survived']

        return X, y

################ Encoding functions ###############
    def male_encode(self, df) :
        for col in df.select_dtypes(object):
            df[col] = df[col].map(self.dic_param)
        
        return df
    
    def female_encode(self, df) :
        for col in df.select_dtypes(object):
            df[col] = df[col].map(self.dic_param)
        
        return df

################ Imputing functions ###############
    def male_impute(self, df) :
        return df.dropna()
    
    def female_impute(self, df) :
        return df.dropna()
    