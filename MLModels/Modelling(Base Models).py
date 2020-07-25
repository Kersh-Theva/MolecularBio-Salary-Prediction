#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:47:10 2020

@author: kershtheva
"""

#ML for base models without job description or job title for minSalary, maxSalary and mean Salary 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline

#Read in csv from post-EDA analysis
glassdoorDB = pd.read_csv("/Users/kershtheva/Desktop/MolecularBioSalary_Prediction/ExploratoryDataAnalysis/glassdoorDB_postEDA.csv")

#Take out target: Mean Salary
yMean = glassdoorDB['Mean Salary'].values
yMin = glassdoorDB['Min Salary'].values
yMax = glassdoorDB['Max Salary'].values

gDBTrain = glassdoorDB.drop(columns=['Mean Salary','Unnamed: 0', 'Job Title', 'Job Description', 'Min Salary', 'Max Salary'], axis=1)

#Standardize data
X = (gDBTrain-gDBTrain.mean())/gDBTrain.std()
#------------------------------------------------------------

#Import sklearn libraries
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge 

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.feature_selection import SelectFromModel

#------------------------------------------------------------

#Make train and test data, split 90:10. 
x_train, x_test, y_trainMean, y_testMean = train_test_split(X, yMean, test_size = 0.1)
x_train, x_test, y_trainMin, y_testMin = train_test_split(X, yMin, test_size = 0.1)
x_train, x_test, y_trainMax, y_testMax = train_test_split(X, yMax, test_size = 0.1)

#------------------------------------------------------------

def lassoSearch(alphaMin, alphaMax, stride, features, target, scoring='neg_mean_absolute_error', cv=3):
    '''
    Uses a range of alpha values to obtain cross_val_scores for Lasso Regression. 
    
    Arguments:
    alphaMin -- Mininum alpha value (assume int)
    alphaMax -- Maximum alpha value (assume int)
    stride -- stride for range function from min. - max. alpha
    features -- labelled features to use for training
    target - target value to predict using fit model 
    cv -- Number of folds for cross validation (default: 3)

    Output: 
    
    alphaPlot -- A plot of negative average MSE value from cross_val_score
    bestAlpha -- The alpha value and MSE value closest to 0 
    '''
    
    #initialize alpha and error lists for alphaPlot
    alpha = []
    error = []
    
    #search through alpha values from alphaMin to alphaMax with stride=stride
    for i in range (alphaMin,alphaMax, stride):
        newAlpha = i/10
        alpha.append(newAlpha)
        lm_lasso = Lasso(newAlpha)
        error.append(np.mean(cross_val_score(lm_lasso, features, target, scoring=scoring, cv=cv)))
      
    #Plot -average MSE value vs. error 
    err = list(zip(alpha,error))
    plt.plot(alpha,error)
    plt.title("Mean MSE vs. Alpha for Lasso Regression")
    plt.ylabel("-Mean MSE")
    plt.xlabel("Alpha Value")

    #Return best alpha value
    errorDB = pd.DataFrame(data=err, columns = ['alpha', 'error'])
    return errorDB[errorDB.error == max(errorDB.error)].iloc[0]

#Invoke lassoSearch function
lassoSearch(1, 100, 1, x_train, y_trainMean, scoring='neg_mean_absolute_error', cv=5)
lassoSearch(1, 100, 1, x_train, y_trainMin, scoring='neg_mean_absolute_error', cv=5)
lassoSearch(1, 100, 1, x_train, y_trainMax, scoring='neg_mean_absolute_error', cv=5)

#------------------------------------------------------------

def modelFit (modelList, features, target, scoring='neg_mean_absolute_error', cv=5, polynomial=False, degree = 2):
    '''
    Fits a list of base models using cross_val_score and returns the output in a sorted dataFrame.
    May optionally perform polynomial transform and linear regression on the results with default degree 2
    
    Arguments:
    modelList -- A list of base models to use with cross_val_score 
    features -- labelled features to use for training
    target - target value to predict using fit model 
    cv -- Number of folds for cross validation (default: 5)
    
    Output:
    baseModelDB -- DataFrame of RMSE values for each model of interest with columns model Object and MSE 
    '''
    
    #Initialize mseList and modelName to input into new dataFrame
    mseList = []
    modelNames = []
    
    #Iterate through the models in modelList, getting cross_val_scores for each 
    for model in modelList: 
        
        modelObject = model 
        modelName = str(model)
        modelNames.append(modelName.split('(')[0])
        
        MSE = np.mean(cross_val_score(modelObject, features, target, scoring=scoring, cv=cv))
        mseList.append(MSE)
        
    #If user desires polynomial transform information, perform polynomial linear regression and append to the list
    if polynomial == True: 
        
        #Name the model based on the degree (default is 2)
        modelName = 'Polynomial (Degree: %s)' %degree
        modelNames.append(modelName)
        
        #initialize polyLinReg object 
        polyLinReg = LinearRegression()
        polynomial_features= PolynomialFeatures(degree=degree)

        #fit_transform with degree = degree (default 2)
        x_poly = polynomial_features.fit_transform(features)
        
        #cross_val_score with polyLinReg on transformed features
        MSE = np.mean(cross_val_score(polyLinReg, x_poly, target, scoring='neg_mean_absolute_error', cv=5))
        mseList.append(MSE)

    #Create dataframe with modelName and mseList data
    return pd.DataFrame(zip(modelNames, mseList), columns=['Models', 'Mean MAE']).sort_values(by='Mean MAE', ascending = False)

#Invoke modelFit function on five models (including polynomial linear regression) 
modelList = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor()]
baseModelsMean = modelFit (modelList, x_train, y_trainMean, scoring='neg_mean_absolute_error', cv=5, polynomial=True, degree=3)
baseModelsMin = modelFit (modelList, x_train, y_trainMin, scoring='neg_mean_absolute_error', cv=5, polynomial=True, degree=3)
baseModelsMax = modelFit (modelList, x_train, y_trainMax, scoring='neg_mean_absolute_error', cv=5, polynomial=True, degree=3)

#------------------------------------------------------------

def modelTest(modelList, trainFeatures, trainTarget, testFeatures, testTarget, polynomial=False, degree=2):
    '''
    Fits a list of base models to testFeatures and predicts Target. Then compares to actual value and calculates prediction error. 
    
    Arguments:
    modelList -- A list of base models to use with cross_val_score 
    trainFeatures -- labelled features to use for training
    trainTarget - target value to predict using fit model 
    testfeatures -- labelled features from test data to use for training
    testTarget - target value to predict from test data using fit model  
    
    Output:
    baseTestDB -- A dataFrame with model name, average error and accuracy for each base model 
    
    '''
    
    errorList = []
    accuracyList = []
    modelNames = []
    
    for model in modelList:
        
        #save model name 
        modelName = str(model)
        modelNames.append(modelName.split('(')[0])
        
        #get predictions from fit model  
        model.fit(trainFeatures, trainTarget)
        predictions = model.predict(testFeatures)
        
        #obtain error values and mean absolute error 
        errors = np.mean(abs(predictions - testTarget))
        errorList.append(round(errors,4))
    
        #Obtain mean absolute percent error and accuracy values 
        mape = 100 * np.mean(errors / testTarget)
        accuracy = 100 - mape
        accuracyList.append(round(accuracy,2))
        
    #If user desires polynomial transform information, perform polynomial linear regression and append to the list (Debug)
    if polynomial == True:
        
        #Name the model based on the degree (default is 2)
        modelName = 'Polynomial (Degree: %s)' %degree
        modelNames.append(modelName)
        
        #initialize polyLinReg object 
        polyLinReg = LinearRegression()
        polynomial_features= PolynomialFeatures(degree=degree)

        #fit_transform with degree = degree (default 2)
        x_poly = polynomial_features.fit_transform(trainFeatures)
        polyLinReg = polyLinReg.fit(x_poly, trainTarget)

        #Predict test data 
        xTestPoly = polynomial_features.fit_transform(testFeatures)
        y_poly_pred = polyLinReg.predict(xTestPoly)

        #obtain error values and mean absolute error 
        errors = np.mean(abs(y_poly_pred - testTarget))
        errorList.append(round(errors,4))

        #Obtain mean absolute percent error and accuracy values 
        mape = 100 * np.mean(errors / testTarget)
        accuracy = 100 - mape
        accuracyList.append(round(accuracy,2))

    
    return pd.DataFrame(zip(modelNames, errorList, accuracyList), columns=['Models', 'MAE', 'Accuracy']).sort_values(by='Accuracy', ascending = False)

#Invoke modelTest function on five models (including polynomial linear regression) 
baseTestMean = modelTest(modelList, x_train, y_trainMean, x_test, y_testMean, polynomial=False, degree=3)
baseTestMin = modelTest(modelList, x_train, y_trainMin, x_test, y_testMin, polynomial=False, degree=3)
baseTestMax = modelTest(modelList, x_train, y_trainMax, x_test, y_testMax, polynomial=False, degree=3)

baseTrainMean = modelTest(modelList, x_train, y_trainMean, x_train, y_trainMean, polynomial=False, degree=3)
baseTrainMin = modelTest(modelList, x_train, y_trainMin, x_train, y_trainMean, polynomial=False, degree=3)
baseTrainMax = modelTest(modelList, x_train, y_trainMax, x_train, y_trainMean, polynomial=False, degree=3)
 

#------------------------------------------------------------ 