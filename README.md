# Molecular Biologist Salary Prediction: Project Overview
- Created a tool to estimate salaries for jobs in molecular biology (best model accuracy: 60%) to help biologists negotiate income. 
- Scraped over 1000 job listings from Glassdoor using python and the Selenium scraper tool 
- Engineered features from the text of each job description and job title 
- Optimized regression model and measured variance and bias compared to human performance
- Productionized model and implemented a web app using Heroku (flask-ml-api-mbiosalary)

## Code and Resources Used 
**Python Version:** 2.7.16 and 3.8.3 <br>
**Packages:** pandas, numpy, sklearn, matplotlib, selenium, flask, json, pickle <br>
**For Web Framework Requirements:** pip install -r requirements.txt <br>
**Selenium scraper tool:** https://github.com/arapfaik/scraping-glassdoor-selenium (credit to arapfaik) <br>
**Scraper article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905 <br>
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2 <br>

## Web Scraping
Made a few changes to the Selenium code (referenced above) to scrape 1200 "Molecular Biology" job postings from glassdoor.com. These jobs came from all over the US. With each job, we got the following attributes: 

- Job Title
- Min. and Max. Salary Estimate
- Job Description 
- Rating
- Company
- Company Location
- Company headquarters
- Company Size
- Company Founding Year
- Type of Ownership 
- Industry
- Sectory
- Company Annual Revenue
- Competitors 

## Data Cleaning
After scraping the data, I cleaned it up to make it more usable. I made the following changes to create the final variables: 

- Parsed numeric data out of salary
- Removed rows without salaries 
- Parsed company rating out of company column
- Parsed state information from job location and headquarters (left it out of the model)
- Transformed founded data into the age of the company
- Made columns for:
  - Mention of PhD in job description
  - Job location in the same state as HQ (SamePlace)
- Simplified job title
- Manually inspected and removed jobs that were not relevant to molecular biology

## EDA 
![Companies](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/Top10-01.png)<br>
**Figure 1. The top companies hiring right now.** The companies hiring the most in our job listing database are almost all involved in COVID-related research as of July 2020, including Moderna, AstraZeneca and Takeda. <br>
![States](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/JobsvsState.png)<br>
**Figure 2. The states with the most molecular biology job listings.** California, Massachussettes and New York lead the way with the majority of job listings in our database. <br>
![Company Rating](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/RatingvCompanyAge.png)<br>
**Figure 3. The relationship between company rating and company age.** Company ratings have a large variance for young companies, but tend to converge on a mean of ~3.7 <br>
![Correlation Heatmap](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/CorrelationHeatmap.png)<br>
**Figure 4. Pearson correlations between key features (after data cleaning phase 1) and the mean salary** Most of our features correlate weakly with the Mean Salary. The highest positive correlation can be seen between Company Age, Company Revenue and the Mean Salary of the position. <br>

## Model Building
First, the post-EDA cleaned data was normalized and split into train and test sets with a test size of 10%. I screened six different models and evaluated them using mean absolute error and accuracy (1 - mean absolute percent error). I chose these errors because it would make the resulting accuracy relatively interpretable and outliers don't have a strong effect on this type of error compared to root mean squared error (RMSE). 

The following models were tried: 
- **Multiple Linear Regression** - Baseline for the model
- **Lasso Regression** - Because of sparseness in some features, I expected regularization to improve the model. 
- **Multiple Polynomial Regression (degree 2)** - Testing for improvements in the model with nonlinearity. 
- **Random Forest** - Used for the robustness of the model against sparse variables. 
- **Gradient boosted Random Forest** - Gradient boosting may help improve weak learning model ensembles. 
- **Adaboost Random Forest** - Another boosting method to help improve weak learning model ensembles.

After training ML models without a job description (models names start with prefix "Base"), I found that the best train accuracy was 49% on the mean salary. In comparison, the baseline human performance was 71% on the mean salary. To improve the model, I sought to change the features used to train the model by identifying keywords in the job description and job title that might be relevant for salary prediction (e.g. min. and max. required years of experience). These model names have the prefix "Description". 

![ML Model Summary](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/MLModels/ModelImprovements.svg)<br>
**Figure 5. Comparing machine learning models for minimum, maximum and mean salary.** Comparisons of the best regression model for a basic training set without natural language processing ("Base"), with processing ("Description") and the prediction of minimum salary ("Min."), mean salary ("Mean") and max salary ("Max"). Each model was scored based on the mean absolute error (MAE) and accuracy score (1-mean absolute percent error) for the test data. Mean salary was chosen for further optimization because of its potential utility for the user. After further optimization, mean salary model had 56.7% accuracy and the lowest variance of all tested models. <br>
![Feature Importances](https://github.com/Kersh-Theva/MolecularBio-Salary-Prediction/blob/master/MLModels/featureImportance.svg) <br>
**Figure 6. Plotting the 10 most important features for the optimal machine learning model.** The features from the optimized model that had the top 10 highest weights for mean salary prodiction were plotted. Company gae, company rating and max. experience (an engineered feature) had the greatest importance. 

|             | Human Performance | Base Model - Bias | Base Model - Variance | Description Model - Bias | Description Model - Variance |
|-------------|-------------------|-------------------|-----------------------|--------------------------|------------------------------|
| Min. Salary | 64%               | 20%               | -8%                   | 26%                      | -13%                         |
| Max. Salary | 74%               | 16%               | 18%                   | 21%                      | -14%                         |
| Mean Salary | 71%               | 22%               | 5%                    | 19%                      | -4%                          | <br>


**Table 1. Error Analysis Chart.** Analysis of the bias and variance for min. salary, max. salary and mean salary predictions with ML model and human performance.

## Strategies for Improvement 
1. Changing the target variable to a range as opposed to a specific number. 
2. Training with more data to improve variance or the learning rate for bias improvements. Change the regression model algorithm to improve both. 
3. Show the most important features for baseline model to compare to 'description' model. 
4. Add location data to ML model and see if it improves accuracy. 

Debugs required: Sort dict keys from JSON input parameters, determine reason for negative variance values, clean up folders a bit more (e.g. remove DSP1 Repo)
