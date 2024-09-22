import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import json

class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, numeric_features=None):
        self.class_probabilities = {}
        self.feature_probabilities = {}
        self.numeric_features = numeric_features
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = y.unique()
        # take from X numeric features
        X_num = X[self.numeric_features]
        #convert type to int
        X_num = X_num.astype(int)
        #calculate the probability of each class
        class_counts = y.value_counts()
        self.class_probabilities = class_counts/sum(class_counts)
        #calculate the probability of each feature for each class using gaussian naive bayes
        for feature in X_num.columns:
            self.feature_probabilities[feature] = {}
            for label in class_counts.index:
                #calculate the mean and standard deviation of each numeric feature for each class
                mean = X_num.loc[y==label, feature].mean()
                std = X_num.loc[y==label, feature].std()
                self.feature_probabilities[feature][label] = {'mean': mean, 'std': std}

        #take from X categorical features and numeric features
        X_cat = X.drop(self.numeric_features, axis=1)
        #count the number of instances of each class
        class_counts = y.value_counts()
        #calculate the probability of each class
        self.class_probabilities = class_counts/sum(class_counts)
        #calculate the probability of each feature for each class
        for feature in X_cat.columns:
            self.feature_probabilities[feature] = {}
            for category in X_cat[feature].unique():
                self.feature_probabilities[feature][category] = {}
                for label in class_counts.index:
                    count = X_cat.loc[(y==label) & (X_cat[feature]==category)].shape[0]
                    c_count = class_counts[label]
                    #if count = 0 add laplace smoothing
                    if count == 0:
                        count += 1
                        c_count += len(X_cat[feature].unique())
                    self.feature_probabilities[feature][category][label] = count/c_count
        return self.feature_probabilities
    
    def predict_proba(self, X):
        X_cat = X.drop(self.numeric_features, axis=1)
        X_num = X[self.numeric_features]
        predictions = []
        for index, row in X.iterrows():
            prediction = {}
            for label in self.class_probabilities.index:
                class_probability = self.class_probabilities[label]
                for feature in X_cat.columns:
                    class_probability *= self.feature_probabilities[feature][row[feature]][label]
                #calculate gaussian probability for numeric columns
                for feature in X_num.columns:
                    mean = self.feature_probabilities[feature][label]['mean']
                    std = self.feature_probabilities[feature][label]['std']
                    gaussian_probability = (1/(np.sqrt(2*np.pi)*std))*np.exp(-((row[feature]-mean)**2)/(2*std**2))
                    class_probability *= gaussian_probability
                prediction[label] = class_probability
            #divide by the sum of all probabilities to get a value between 0 and 1
            prediction_sum = sum(prediction.values())
            for label in prediction:
                prediction[label] /= prediction_sum  
            predictions.append(prediction)
            #remove the label and turn into an array containing an array of probabilities for each label
        return np.array([list(p.values()) for p in predictions])
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        list = np.argmax(probabilities, axis=1)
        #if 0 return no-recurrence-events if 1 return recurrence-events
        return np.array([self.classes_[i] for i in list])
    
    def get_params(self, deep=True):
        return {'numeric_features': self.numeric_features}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
