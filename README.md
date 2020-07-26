# Molecular Biologist Salary Prediction
- Created a tool to estimate molecular biologist salaries to help young researchers negotiate their income. 
- Scraped over 1000 job listings from Glassdoor using python and the Selenium scraper tool 
- Engineered features from the text of each job description and job title 
Things I'd like to add: Make web app, using OOP to convert input data to x_features

## Code and Resources Used 
**Python Version:** 2.7.16 <br>
**Packages:** pandas, numpy, sklearn, matplotlib, selenium <br>
**For Web Framework Requirements:** pip install -r requirements.txt (not ready yet) <br>
**Selenium scraper tool:** https://github.com/arapfaik/scraping-glassdoor-selenium (credit to arapfaik)

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

## Data Cleaning - Phase 1
After scraping the data, I cleaned it up to make it more usable. I made the following changes to create the final variables: 

- Parsed numeric data out of slaary
- Determined approximate annual wage from hourly wages, assuming full time work
- Removed rows without salaries 
- Parsed state information from job location and headquarters
- Transformed founded data into the age of the company
- Made columns for:
  - PhD mentions
  - Job location in the same state as HQ
- Simplified job title

## EDA 
![Companies](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/Top10-01.png)<br>
**Figure 1. The top companies hiring right now.** The companies hiring the most in our job listing database are almost all involved in COVID-related research as of July 2020. <br>
![States](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/JobsvsState.png)<br>
**Figure 2. The states with the most molecular biology job listings.** California, Massachussettes and New York lead the way with the majority of job listings in our database. <br>
![Company Rating](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/RatingvCompanyAge.png)<br>
**Figure 3. The relationship between company rating and company age.** Company ratings have a large variance for young companies, but tend to converge on a mean of ~3.7 <br>
![Correlation Heatmap](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/CorrelationHeatmap.png)<br>
**Figure 4. Pearson correlations between key features and the mean salary** Most of our features correlate weakly with the Mean Salary. The highest positive correlation can be seen between Company Age, Company Revenue and the Mean Salary of the position. <br>

**Figure 5. Looking for useful words for feature engineering** (coming soon.)

## Machine Learning Models 
![ML Model Summary](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/MLModels/ModelImprovements.svg)<br>
**Figure 6. Comparing machine learning models for minimum, maximum and mean salary.** Comparisons of the best regression model for a basic training set without natural language processing ("Base"), with processing ("Description") and the prediction of minimum salary ("Min."), mean salary ("Mean") and max salary ("Max"). Each model was scored based on the mean absolute error (MAE) and accuracy score (1-mean absolute percent error) for the test data. Mean salary was chosen for further optimization because of its potential utility for the user. After further optimization, mean salary model had 10% improved accuracy and 50% improved MAE:accuracy ratio compared to the base model. <br>
![Feature Importances](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/MLModels/featureImportance.svg) <br>
**Figure 7. Plotting the 10 most important features for the optimal machine learning model.** 

## Productionized Model
Coming soon:
1. Scraper (**done**)
2. Cleanup (**done**)
3. EDA (Mao, Mean Salary vs. Company Age, Revenue, wordFinder) 
4. ML2, ML3 (RMSE, Accuracy, Feature importance) (**done**)
5. Productionize

(Need to fix Figure 7, Figure 5, and Figure 4., added ipynb for postEDA/secondcleanup)

Future projects: 
1. **Classification** with tumor mutation data and **neural networks** using Kaggle data 
2. **Hierarchical clustering** of lab evolution data using **C++**
3. **Metabolic modelling** for pathway building for target products with **distributed computing**

Need to show:
1. Classification 
2. Regression **X**
3. Clustering 
4. C++
5. Distributed computing
6. Neural networks
7. Containerized **X**
8. OOP **X**
9. Data scraping **X**
10. Error analysis **X**
11. Metabolic modelling 

Things to consider:
1. Productionizing the model so that it can accept databases and return outputs in batch. 
2. Analysing human performance as finding the value within a range or training ensemble models for the mean
3. Training with more data, changing the regression model algorithm
