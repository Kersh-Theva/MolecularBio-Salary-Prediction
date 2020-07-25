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
glassdoorDB = pd.read_csv("/Users/kershtheva/Desktop/Data Science Projects/DS Project 1 Repo/gDB(Cleanup2).csv")

#Take out target: Mean Salary
yMean = glassdoorDB['Mean Salary'].values
yMin = glassdoorDB['Min Salary'].values
yMax = glassdoorDB['Max Salary'].values

gDBTrain = glassdoorDB.drop(columns=['Mean Salary','Unnamed: 0', 'Min Salary', 'Max Salary'], axis=1)

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
        MSE = np.mean(cross_val_score(polyLinReg, x_poly, target, scoring=scoring, cv=5))
        mseList.append(MSE)

    #Create dataframe with modelName and mseList data
    return pd.DataFrame(zip(modelNames, mseList), columns=['Models', 'Mean MAE']).sort_values(by='Mean MAE', ascending = False)

#Invoke modelFit function on five models (including polynomial linear regression) 
modelList = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor()]
descriptModelsMean = modelFit (modelList, x_train, y_trainMean, scoring='neg_mean_absolute_error', cv=5, polynomial=True, degree=3)
descriptModelsMin = modelFit (modelList, x_train, y_trainMin, scoring='neg_mean_absolute_error', cv=5, polynomial=True, degree=3)
descriptModelsMax = modelFit (modelList, x_train, y_trainMax, scoring='neg_mean_absolute_error', cv=5, polynomial=True, degree=3)

#Data suggests it would be best to go with Adaboost

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
        
    #If user desires polynomial transform information, perform polynomial linear regression and append to the list (NEED TO FIX)
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
descriptTestMean = modelTest(modelList, x_train, y_trainMean, x_test, y_testMean, polynomial=False, degree=3)
descriptTestMin = modelTest(modelList, x_train, y_trainMin, x_test, y_testMin, polynomial=False, degree=3)
descriptTestMax = modelTest(modelList, x_train, y_trainMax, x_test, y_testMax, polynomial=False, degree=3)

descriptTrainMean = modelTest(modelList, x_train, y_trainMean, x_train, y_trainMean, polynomial=False, degree=3)
descriptTrainMin = modelTest(modelList, x_train, y_trainMin, x_train, y_trainMean, polynomial=False, degree=3)
descriptTrainMax = modelTest(modelList, x_train, y_trainMax, x_train, y_trainMean, polynomial=False, degree=3)
#------------------------------------------------------------

#Randomized gridsearch 

def randomSearch(estimator, parameters, trainFeatures, trainTarget, testFeatures, testTarget, n_iterations = 100, cv=3):
    
    '''
    Perform RandomizedSearch using provided parameters and estimators and output the parameters producing bestoutput 
    
    Arguments:
    Estimator -- The chosen machine learning model 
    Parameters -- A dictionary of parameters to use with randomizedsearchCV 
    n_iterations -- Number of searches to perform through parameters (default: 100)
    trainFeatures -- labelled features to use for training
    trainTarget -- target value to predict using fit model 
    testfeatures -- labelled features from test data to use for training
    testTarget -- target value to predict from test data using fit model  
    n_iterations -- Number of iterations of randomSearch to perform (default: 100)
    cv -- Number of folds for cross validation (default: 3)
    
    Outputs:
    Print statement with the best parameters found for the model
    bestRandomDB -- A dataframe with the output from modelTest for the best model found with randomsearchCV
    bestRandom -- best model from randomSearch
    '''
    
    modelRandom = RandomizedSearchCV(estimator = estimator, 
                                   param_distributions = parameters, 
                                   n_iter = 100, 
                                   cv = 3, 
                                   random_state=42, 
                                   n_jobs = -1 #use all available cores
                                   )
        
    # Fit the random search model
    modelRandom.fit(trainFeatures, trainTarget)
    
    #find the best estimator for model testing
    bestRandom = modelRandom.best_estimator_
    
    #Print the best parameters
    print(modelRandom.best_params_)
    
    #Store the database returned by modelTest in bestRandom 
    bestRandomDB = modelTest([bestRandom], trainFeatures, trainTarget, testFeatures, testTarget, polynomial=False, degree=2)
    return bestRandomDB

#Gradient boost random search
gboostRandomGrid = {'criterion': ['mae'],
                    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
                    'max_features': ['auto', 'sqrt', 'log2']
                    }

gBoost = GradientBoostingRegressor() 

#Accuracy on test set
bestRandomGradientDB = randomSearch(gBoost, gboostRandomGrid, x_train, y_trainMean, x_test, y_testMean, n_iterations = 30, cv=3)

#Accuracy on train test
bestRandomBoostTrainDB = randomSearch(gBoost, gboostRandomGrid, x_train, y_trainMean, x_train, y_trainMean, n_iterations = 30, cv=3)

#BestBoost
#{'n_estimators': 100, 'max_features': 'log2', 'criterion': 'mae'}
#------------------------------------------------------------

def gridSearch(estimator, parameters, trainFeatures, trainTarget, testFeatures, testTarget, cv=3):
   
    '''
    Perform GridSearch using provided parameters and estimators and output the parameters producing bestoutput 
    
    Arguments:
    Estimator -- The chosen machine learning model 
    Parameters -- A dictionary of parameters to use with randomizedsearchCV 
    trainFeatures -- labelled features to use for training
    trainTarget -- target value to predict using fit model 
    testfeatures -- labelled features from test data to use for training
    testTarget -- target value to predict from test data using fit model  
    cv -- Number of folds for cross validation (default: 3)
    
    Outputs:
    Print statement with the best parameters found for the model
    bestGrid - best model from gridSearch 
    '''
    
    #Gridsearch with provided estimator and parameters 
    gridSearch = GridSearchCV(estimator, parameters, cv = 3, n_jobs = -1, verbose = 2)

    # Fit the grid search model
    gridSearch.fit(trainFeatures, trainTarget)
    
    #Get the best grid 
    bestGrid = gridSearch.best_estimator_

    #Print out best parameters for user
    print(gridSearch.best_params_)
    return bestGrid
    

gboostGridSearch = {'max_depth': [int(x) for x in np.linspace(start = 3, stop = 10, num = 6)],
              'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
              }

gBoost = GradientBoostingRegressor()

#Try finding best model with gridSearch and focused parameters
bestBoostModel = gridSearch(gBoost, gboostGridSearch, x_train, y_trainMean, x_test, y_testMean, cv=3)

#Store the database returned by modelTest in bestBoostDB
bestBoostGridDB = modelTest([bestBoostModel], x_train, y_trainMean, x_test, y_testMean, polynomial=False, degree=2)
#------------------------------------------------------------
 
def featureExtract(model, trainFeatures, trainTarget, topFeatures, fileDirectory): 
    '''
    Determine the importance of features from the best model. Save figure to file
    
    Arguments:
    model -- Best model from modelFit, modelTest, randomSearch and gridSearch
    trainFeatures -- labelled features to use for training
    trainTarget - target value to predict using fit model    
    topFeatures -- number of features to show - assumes int
    fileDirectory - directory path to save file 

    Outputs:
    featureGraph = Plot of feature importances and their weights for the best model
    '''
    
    #Use SelectFromModel to extract feature information from selected weights  
    #select = SelectFromModel(model)
    model.fit(trainFeatures, trainTarget)
    #select.get_support() #.estimator_

    #Make a series of feature importances for each provided feature
    featImportance = pd.Series(model.feature_importances_, index = trainFeatures.columns).sort_values(ascending=False)
    featureGraph = featImportance.nlargest(topFeatures).plot(kind = 'bar', figsize=(5,5), title = "Feature Importances",
                                           colormap = 'Spectral')
    
    plt.ylabel('Feature Importance Score')
    plt.savefig(fileDirectory)
    
    #If feature column indices are of interest
    #indices = np.argsort(featImportance)[::-1]
    
    #Plot this data
    #featImportance.plot(kind='bar', title='Feature Importances')
    #plt.bar(range(X.shape[1]), featImportance[indices],
    #color="r", align="center")
    
    #Modify aspects of the plot to make it look better
    #plt.xticks(range(X.shape[1]), indices)
    #plt.xlim([-1, X.shape[1]])
    #plt.title("Feature importances")

    #plt.figure(figsize=(3,4))
    #plt.show()

bestModel = GradientBoostingRegressor(n_estimators=100, max_features='log2', criterion='mae')
featureExtract(bestModel, x_train, y_trainMean, 10, "/Users/kershtheva/Desktop/MolecularBioSalary_Prediction/featureImportance.svg")

#------------------------------------------------------------
 
import seaborn as sns

def modelDisplay(modelList):
    '''
    Show the train accuracy, test accuracy and accuracy differences (variance) for the best model in each DB in modelList.
    Display values in three bar plots.
        
    Arguments:
    modelList -- List of databases to concatenate and display data - assumes list of dataframes, 7 models provided

    Returns:
    Barplots for each of the three metrics indicated above. Save figure to hard-coded file path. 
    Database containing each of the values that was plotted. 
    '''
    #Concatenate pd models
    categoryList = ['Base - Min.' ,'Description - Min.', 
                    'Base - Max.', 'Description - Max.', 
                    'Base - Mean', 'Description  - Mean',  'Optimized - Mean']
    
    #Initialize lists
    trainAccuracyValues = []
    testAccuracyValues = []
    varianceList = []
    counter = 1
    
    #Itereate through models
    for model in modelList:
        
        #If it's a database for test accuracies
        if counter%2 == 0:
        
            #Find max value of accuracy and append all values of that row to relevant lists
            maxix = model.Accuracy.idxmax(axis=0) 
            
            testAccuracyValue = model.loc[maxix, 'Accuracy'] 
            testAccuracyValues.append(testAccuracyValue)
        
        #If it's a database for train accuacies
        else: 
            
            #Find max value of accuracy and append all values of that row to relevant lists
            maxix = model.Accuracy.idxmax(axis=0) 
            
            trainAccuracyValue = model.loc[maxix, 'Accuracy'] 
            trainAccuracyValues.append(trainAccuracyValue)
        
        counter += 1 
        
    #Iterate through indices of the testAccuracy list and compare to train accuracy
    for ix in range(len(testAccuracyValues)): 
        
        variance = trainAccuracyValues[ix] - testAccuracyValues[ix]
        varianceList.append(variance)
        
    #Create dataframe with data
    fullList = list(zip(categoryList, trainAccuracyValues, testAccuracyValues, varianceList))
    maxDB = pd.DataFrame(fullList, columns=['Model', 'Train Accuracy (%)', 'Test Accuracy (%)', 'Variance'])
    
    #Plot data in 2 plots with SNS barplot..
    sns.set(style="ticks", palette="pastel")
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    
    #Accuracy Data
    sns.barplot(data=maxDB, x='Model', y = 'Train Accuracy (%)', palette="rocket", ax=ax1)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel('Train Accuracy (%)')
    #plt.savefig("/Users/kershtheva/Desktop/MolecularBioSalary_Prediction/Accuracy.pdf")

    #MAE Data
    sns.barplot(data=maxDB, x='Model', y = 'Test Accuracy (%)', palette = "rocket", ax=ax2)
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel('Test Accuracy (%)')
    
    #Ratio Data
    sns.barplot(data=maxDB, x='Model', y = 'Variance', palette = "rocket", ax=ax3)
    ax3.axhline(0, color="k", clip_on=False)
    ax3.set_ylabel('Variance (Train-Test)')

    # Finalize the plot
    sns.despine(bottom=False)
    plt.setp(f.axes)
    plt.tight_layout(h_pad=2)
    plt.savefig("/Users/kershtheva/Desktop/MolecularBioSalary_Prediction/MLModels/ModelImprovements.svg")
    
    return maxDB
    
#Combining the databases together and plotting them 
bestRandomBoostTestDB=bestRandomGradientDB
modelList = [baseTestMin, 
             baseTrainMin, 
             descriptTestMin, 
             descriptTrainMin, 
             baseTestMax, 
             baseTrainMax,
             descriptTestMax, 
             descriptTrainMax,
             baseTestMean, 
             baseTrainMean,
             descriptTestMean, 
             descriptTrainMean,
             bestRandomBoostTestDB,
             bestRandomBoostTrainDB]

modelDisplay = modelDisplay(modelList)    

modelDisplay.to_csv("/Users/kershtheva/Desktop/MolecularBioSalary_Prediction/MLModels/ModelImprovements.csv")
    
    
    
    
    
    
    
    