# Molecular Biologist Salary Prediction

- Created a tool to estimate molecular biologist salaries to help molecular biology researchers negotiate their income. 
- Scraped over 1000 job listings from Glassdoor using python and the Selenium scraper tool 
- Engineered features from the text of each job description and job title to determine whether "PhD" was included in the position and classified job titles

## Code and Resources Used 
**Python Version:** 2.7.16 <br>
**Packages:** pandas, numpy, sklearn, matplotlib, selenium <br>
**For Web Framework Requirements:** pip install -r requirements.txt (not ready yet) <br>
**Selenium scraper tool:** https://github.com/arapfaik/scraping-glassdoor-selenium (credit to arapfaik)

## Web Scraping

Made a few changes to the Selenium code (referenced above) to scrape 1200 "Molecular Biology" job postings from glassdoor.com. These jobs came from all over the US. With each job, we got the following attribute: 

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
  
![Companies](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/Top10-01.png)

**Figure 1. The top companies hiring right now.** The companies hiring the most in our job listing database are almost all involved in COVID-related research as of July 2020. 

![States](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/JobsvsState.png)

**Figure 2. The states with the most molecular biology job listings.** California, Massachussettes and New York lead the way with the majority of job listings in our database. 

![Company Rating](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/RatingvCompanyAge.png)

**Figure 3. The relationship between company rating and company age.** Company ratings have a large variance for young companies, but tend to converge on a mean of ~3.7

![Correlation Heatmap](https://github.com/Kersh-Theva/MolecularBioSalary_Prediction/blob/master/ExploratoryDataAnalysis/CorrelationHeatmap.png)

**Figure 4. Pearson correlations between key features and the mean salary** Most of our features correlate weakly with the Mean Salary. The highest positive correlation can be seen between Company Age, Company Revenue and the Mean Salary of the position. 
