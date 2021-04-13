#%% CREATE ADABOOST FROM SCRATCH
### Alvaro Corrales
### April 2021 (work in progress)

# Imports
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Helper functions
def compute_error(w_i, y, y_pred):
    '''
    Calculate the error rate of a weak classifier m. Parameters:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier
    
    Note that all arrays should be the same length
    '''
    return (sum(w_i * (y == y_pred).astype(int)))/sum(w_i)

def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Parameters:
    error: error rate from weak classifier m
    '''
    return np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):
    ''' 
    Update individual weights w_i after a boosting iteration. Parameters:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate y_pred
    '''  
    return w_i * np.exp(alpha * (y == y_pred).astype(int))

# Define AdaBoost class
class AdaBoost:
    
    def __init__(self):
        # self.w_i = None
        self.alphas = []
        self.G_M = []
        self.M = None

    def fit(self, X, y, M = 100):
        '''
        Fit model. Parameters:
        X: independent variables
        y: target variable
        M: number of boosting rounds. Default is 100
        '''

        self.M = M

        # Iterate over M weak classifiers
        for m in range(0, M + 1):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                update_weights(w_i, alpha_m, y, y_pred)
            
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = compute_error(w_i, y, y_pred)

            # (c) Compute alpha
            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)


    def predict(self, X):
        '''
        Predict using fitted model. Parameters:
        X: independent variables
        '''

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # Estimate final predictions
        y_pred = (-1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
      