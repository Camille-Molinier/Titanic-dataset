import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report; plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures

class model_trainer :
     ################### Constructor ###################
    def __init__(self) :
        pass

    #################### Evaluation ###################
    def evaluation (self, model, name, X_train, y_train, X_test, y_test) :
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        plt.figure()
        plot_confusion_matrix(model, X_test, y_test)
        plt.savefig(f'../dat/fig/models/{name} matrix.png')

        N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))
        plt.figure()
        plt.plot(N, train_score.mean(axis=1), label='Train score')
        plt.plot(N, val_score.mean(axis=1), label='Val score')
        plt.legend()
        plt.savefig(f'../dat/fig/models/{name} learning curve')