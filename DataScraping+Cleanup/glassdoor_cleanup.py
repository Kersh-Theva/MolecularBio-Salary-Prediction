#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:00:55 2020

@author: kershtheva

Cleaning up glassdoorFullDB:
    - Company_Name: Remove numbers (done) 
    - Competitors: Remove because more than half of them don't have this info;(done)
    - Founded: Rename to Company_Age (obtain 2020-year), -1 remove (done)
    - Headquarters: Is headquarters as important as Location? remove -1 values, just keep state (done)
    - Industry: Remove industry -1 values (done)
    - Location: Just focus on State information  (done)
    - Rating: No changes needed, just confirm its an float type (done)
    - Revenue: Remove (USD), remove some -1, if it has a + sign, we can remove that, remove $ signs, (done) (used higher # in revenue range)
    - Salary Estimate: Remove Glassdoor Is it hourly? Separate into minimum and maximum, average salary.
    - Sector: How is sector different from industry? (done)
    - Size: Remove employees, Lowest number and to will be removed (done)
    - Type of ownership: Company - Public = Public, Company - Private = Private,(done!)
        Other organization = Other, Nonprofit organization = Non-profit

Feature engineering suggestions: 
    - HQ and location are in different places? 
    - Average salary 
    - Job description: Let's hold on to it and see if we can get some insights
    - Job title: Look at most common words and see what I can do with them 
-     
"""

#Company name
glassdoorFullDB = pd.read_csv("/Users/kershtheva/Desktop/MolecularBioSalary_Prediction/glassdoorFullDB.csv")
glassdoorFullDB.rename(columns = {'Company Name': 'Company_Name'}, inplace = True)
glassdoorFullDB['Company_Name'] = glassdoorFullDB['Company_Name'].apply(lambda x: x.split('\n')[0])

#Competitors
glassdoorFullDB.drop(columns = ['Competitors'], inplace=True)

#Salary Estimate
glassdoorFullDB['Salary Estimate'] = glassdoorFullDB['Salary Estimate'].apply(lambda x: x.split('(')[0])
glassdoorFullDB['Salary Estimate'] = glassdoorFullDB['Salary Estimate'].apply(lambda x: x.replace('K','').replace('$', ''))
glassdoorFullDB['hourly'] = glassdoorFullDB['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)

glassdoorFullDB['min_salary'] = glassdoorFullDB['Salary Estimate'].apply(lambda x: x.split('-')[0]).astype(int)
glassdoorFullDB['max_salary'] = glassdoorFullDB['Salary Estimate'].apply(lambda x: x.split('-')[1])
glassdoorFullDB['max_salary'] = glassdoorFullDB['max_salary'].apply(lambda x: x.replace('Per Hour',''))
glassdoorFullDB['max_salary'] = glassdoorFullDB['max_salary'].astype(int)
glassdoorFullDB.drop(columns = ['Salary Estimate'], inplace=True)
glassdoorFullDB['Mean Salary'] = (glassdoorFullDB['max_salary'] + glassdoorFullDB['min_salary'])/2

#Headquarters 
glassdoorFullDB = glassdoorFullDB[glassdoorFullDB['Headquarters'] != '-1']
glassdoorFullDB['Headquarters'] = glassdoorFullDB['Headquarters'].apply(lambda x: str(x).split(', ')[1])

#industry
glassdoorFullDB = glassdoorFullDB[glassdoorFullDB['Industry'] != '-1']

#Location
glassdoorFullDB['Location'] = glassdoorFullDB['Location'].apply(lambda x: x.split(', ')[-1])
glassdoorFullDB['Location'] = glassdoorFullDB['Location'].replace('California', 'CA').replace('Illinois', 'IL').replace('United States', 'US')

#for company age
glassdoorFullDB['Company_Age'] = glassdoorFullDB['Founded'].apply(lambda x: 2020-int(x) if x != -1 else 0)
glassdoorFullDB.drop(columns = ['Founded'], inplace=True)

#Rating: check if float
glassdoorFullDB['Rating'] = glassdoorFullDB['Rating'].apply(lambda x: float(x))

#Revenue
glassdoorFullDB = glassdoorFullDB[glassdoorFullDB['Revenue'] != -1]
glassdoorFullDB['Revenue'] = glassdoorFullDB['Revenue'].apply(lambda x: str(x).replace('$', '').replace('(USD)', '').replace('million', ''))

glassdoorFullDB['Revenue']  = glassdoorFullDB['Revenue'].apply(lambda x: str(x).replace('billion', '000').replace('Unknown / Non-Applicable', '0').replace('+', '').replace('Less than ', ''))
glassdoorFullDB['Revenue']  = glassdoorFullDB['Revenue'] .apply(lambda x: x.split('to ')[1] if 'to ' in x else x)
glassdoorFullDB['Revenue']  = glassdoorFullDB['Revenue'] .apply(lambda x: str(x).replace(' ', ''))

#Sector
glassdoorFullDB.drop(columns='Sector', inplace=True)

#Size 
glassdoorFullDB = glassdoorFullDB[glassdoorFullDB['Size'] != -1]
glassdoorFullDB['min_size'] = glassdoorFullDB['Size'].apply(lambda x: str(x).split(' ')[0]).replace('Unknown', '10')

#Type of ownership 
glassdoorFullDB = glassdoorFullDB[glassdoorFullDB['Type of ownership'] != -1]

glassdoorFullDB['Type of ownership'] = glassdoorFullDB['Type of ownership'].apply(lambda x: str(x).split('-')[-1] if '-' in x else x).replace('College / University', 'Academic').replace('Other Organization', 'Other')
glassdoorFullDB['Type of ownership'] = glassdoorFullDB['Type of ownership'].replace('Subsidiary or Business Segment', ' Private').replace('Nonprofit Organization', 'Nonprofit')

glassdoorFullDB.drop(columns='Unnamed: 0', inplace=True)
glassdoorFullDB.drop(columns='Size', inplace=True)

glassdoorFullDB.to_csv("/Users/kershtheva/Desktop/MolecularBioSalary_Prediction/glassdoorFullDBClean.csv")
#--------
