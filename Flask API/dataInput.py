#Purpose of this python file is to create and manipulate user-provided parameters
#as a dictionary-like object to get its attributes and functions. 

import pandas as pd 
import numpy as np

      
def yearsOfExperience(Dict, key, yearsList): 
    '''
    Finds a list of numbers associated with years/years of experience 
    in a job description provided in a dict. 
    
    Arguments: 
    Dict -- Parameter dictionary provided by user
    key -- Key name associated with job description; assumes string 
    yearsList -- List of strings with word to look for 
    
    Returns:
    wordFrame -- 1D DataFrame where each row has a list of words and numbers surrounding the word from yearsList
                Returned dataframe will have ix rows, where ix = number of jobs provided by user (assumes one, for now)
    
    '''
    #Initializing list of lists, where every list of lists element represents a series row
    seriesList = []
    
    #search through every row of the series
    for ix in Dict[key]:  
        
        #split the description into individual words
        paragraph = ix.split()
        
        #sublist for seriesList
        paragraphList = []
        
        #search through terms in yearsList
        for term in yearsList:
            
            indices = (i for i, word in enumerate(paragraph) if word==term)
            
            #Make indexList and add it + 5 word range to the current paragraph list  
            for ind in indices: 
                
                indList = paragraph[ind-5:ind]+paragraph[ind:ind+5]
                paragraphList = paragraphList + indList

        #Append paragraphList to seriesList for everyrow
        seriesList.append(paragraphList)
    
    wordFrame = pd.DataFrame({'Word Range': seriesList})
    return wordFrame
                

def minMaxYears(dataFrame, series, minYears, maxYears): 
    '''
    Finds the minimum and maximum number of years of experience needed for a particular job in a Dataframe 
    
    Arguments: 
    dataFrame -- A dataFrame of interest
    series -- A series name from dataFrame, assumes this is a string
    minYears -- The smallest number of years to look for
    maxYears -- The largest number of years to look for
    
    If there is only one number in the row, we will assume this is the minimum number of years and max years
    
    Returns:
    minmaxDB -- DataFrame with two columns and int type values: Min Experience, Max Experience
    
    '''
    yearsList = list(range(minYears, maxYears+1))
    
    #Initialize lists
    minValList = []
    maxValList = []

    #Iterate through dataframe rows
    for row in dataFrame[series]:
        
        #only look for maxVal when minvalFound becomes true
        minValFound = False
        
        #make all the list items into one string to make iteration simpler
        rowEntry = " ".join(row)
        rowEntry = rowEntry.replace('+','')
    
        #Iterate through yearsList from smallest to largest             
        for year in yearsList:
            
            #set the year to a string to iterate through rowEntry
            yearString = str(year)
            
            if minValFound != True and yearString in rowEntry:
                minVal = year
                minValList.append(year)
                maxVal = year 
                minValFound = True
                
            else:
                continue
        
        if minValFound == False: 
            minValList.append(0)
            maxVal = 0
            
        #Once min is done, look for maxVal
        if minValFound == True:
            for year in range(minVal, len(yearsList)): 
                yearString = str(year)
                
                if yearString in rowEntry: 
                    maxVal = year 

                else:
                    continue
                         
        maxValList.append(maxVal)
    
    #Once all rows are done, make the columns.
    minMaxDB = pd.DataFrame({'Min. Experience': minValList, 'Max. Experience': maxValList})
    return minMaxDB


def phraseCounter(df, series, phraseList, newSeriesName):
    '''
    Takes a list of phrases and returns a 1 or 0 if the phrase is found in the series
    
    Arguments: 
        dataFrame -- dataFrame with series of interest
        series -- name of the series; assumes name is passed as string
        phraselist -- list of phrases; assumes each phrase is a string and lowercase
        newSeriesName -- new series name for series to join with original dfl assumes name is passed as string
        
    Returns:
        Returns extended dataframe with a 1 or 0 in each row depending on whether the phrase is found in the entry
        Will fill NaN values with 0 
    '''
    #Make copy of df to avoid changing original    
    dfCopy = df.copy()
    countList = []
    
    #Iterate through the rows
    for row in dfCopy[series]:
        
        #default value is 0
        row = row.lower()
        counter = 0
      
        #iterate through the phraseList
        for phrase in phraseList: 
            
            #change counter if the phrase is found
            if phrase in row:
                counter = 1
            
            else:
                continue
        
        countList.append(counter)
    
    #Make new series and append it to the provided dataframe with the provided newSeriesName
    newSeries = pd.DataFrame({newSeriesName: countList})
    newDB = df.join(newSeries)
    
    #fill in Na values should there be any and return the new DB
    newDB[newSeriesName] = newDB[newSeriesName].fillna(value=0)
    return newDB

def dictPhraseCounter(df, series, searchDict):
    '''
    Looks through a provided dictionary of keys and values, assuming each value is a list
    
    Input: 
    dataFrame -> Original dataFrame
    series -> The series in the original dataFrame to look for
    searchDict -> Keys are the series Name, and values are the phraseList; can also be just be individual string
    '''
    for key in searchDict.keys():       
        df = phraseCounter(df, series, searchDict[key], key)
    return df


#Get rating, company age, revenue, size, job title, job description
class jobDescriptor(object): 
    
    '''
    A dict-type object where keys are parameter names. 
    Features will be extracted from object attributes to generate feature vector for MBio Salary Prediction model
    
    '''

    def __init__(self, init=None):
        if init is not None:
            self.__dict__.update(init)
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)
    
    def toDataFrame(self, entries=1): 
        '''
        Converts dictionary to dataframe 
        
        Arguments:
        entries -- Number of entries provided by user (default: 1)
        '''
        return pd.DataFrame(self.__dict__, index=range(entries))

    def companyAge(self, endYear=2020):
        '''
        Computes company age from provided company founding age and endYear (default:2020)
        '''
        
        return endYear - self.__getitem__('Company Age')
    
    def minMax(self):
        '''
        Get the minimum and maximum years of experience required for a job
        '''
        
        #list of words to be used to find the years of experience
        yearsList = ['experience','years', ' yr ', ' yrs ']

        #Make dataframe from dictionary
        paramDB = self.toDataFrame()
        
        #Get the neighborhood of words and numbers around each term in yearsList in each row
        wordRange = yearsOfExperience(paramDB, 'Job Description', yearsList)
        
        #Get the minimum and maximum number from the merged list for each row
        return paramDB.join(minMaxYears(wordRange, 'Word Range', 0, 10))
  
        
    def wordHunt(self):
        '''
        Provides a set of terms in descriptionDict and titleDict to search for. 

        Returns a binary value of 0 or 1 for each word found in 

        '''
        
        descriptionDict = {
            'Lead': ['lead'],
            'Manage': ['manage'], 
            'Employees': ['employee'],
            'Genetic': ['genetic'],
            'Biochemistry': ['biochemistry'],
            'RNA': ['rna'],
            'DNA': ['dna'],
            'Protein': ['protein'],
            'Crystallography': ['crystal'], 
            'No Experience': ['0-'],
            'Cancer': ['cancer'], 
            'Immunology': ['immun'], 
            'Therapy': ['therapy'],
            'BSc': ['bachelor', 'b.s.', 'bsc'], 
            'MSc': ['master', 'm.s'],
            'PhD':['phd', 'ph.d'],
            'ML': ['machine learning'],
            'Postdoctoral': ['postdoctor'],
            'Python': ['python'] 
            }
        
        titleDict = {
            'VP': ['vp'],
            'Vice President': ['vice president'],
            'I': [' i'],
            'II': [' ii'],
            'Principal': ['principal'],
            'Manager': ['manage'],
            'Investigator': ['investigat'],
            'Researcher': ['research'],
            'Supervisor': ['supervisor'],
            'Director': ['director'],
            'Associate': ['associate'],
            'Technologist': ['technolo'],
            'Fellow': ['fellow'],
            'Technician': ['technic'],
            'Analyst': ['analyst'],
            'Scientist I': ['scientist i'],
            'Scientist II': ['scientist ii'],
            'Scientist':[ 'scientist'],
            'Senior': ['senior', 'sr.'],
            'Engineer': ['engineer'],
            'QC':['qc', 'quality control'],
            'Virologist': ['virologist']
            }
    
        #Get DB from minmax
        minMaxDB = self.minMax()
        
        #make db with job title 
        titleDB= dictPhraseCounter(minMaxDB, 'Job Title', titleDict)
            
        #make db with job description
        return dictPhraseCounter(titleDB, 'Job Description', descriptionDict)

        
    def transformDict(self): 
        '''
        Make final array of features from dict values, minMax and wordHunt. 
        '''

        #Make dataframe from wordHunt result
        queryDB = self.wordHunt()
        
        #Drop the title and description columns
        queryDB = queryDB.drop(columns=['Job Title', 'Job Description'])

        #The mean values for each of the categories to iterate through
        meanValues = [3.754020, 2686.546078, 46.085294, 3231.455882, 4.362745, 2.060784, 0.217647,
                      0.200000, 0.560784, 0.403922, 0.032353, 0.534314, 0.575490, 0.389216,  0.050980,
                      0.411765, 0.138235, 0.111765, 0.061765, 0.490196, 0.361765, 0.006863, 0.300000,
                      0.639216, 0.400980, 0.008824, 0.406863, 0.095098, 0.073529, 0.043137, .057843,
                      0.010784, 0.002941, 0.008824, 0.277451, 0.089216, 0.258824, 0.031373, 0.003922,
                      0.010784, 0.420588, 0.019608, 0.013725, 0.024510, 0.347059, 0.092157, 0.060784]
        
        #The standard deviation values for each of the categories to iterate through
        stdValues = [0.696573, 3970.506680, 49.696089, 4004.214919, 2.393109, 1.534117, 0.412849,
                     0.400196, 0.496535, 0.490923, 0.177022, 0.499066, 0.494511, 0.487812, 0.220066,
                     0.492394, 0.345316, 0.315231, 0.240846, 0.500149, 0.480747, 0.082597, 0.458482,
                     0.480464, 0.490337, 0.093564, 0.491490, 0.293494, 0.261132, 0.203266, 0.233561,
                     0.103337, 0.054179, 0.093564, 0.447961, 0.285195, 0.438203, 0.174408, 0.062530,
                     0.103337, 0.493896, 0.138716, 0.116406, 0.154701, 0.476268, 0.289389, 0.239051]
        
        #Make a list of all queries after transformation
        queryLists = []
        
        #Make a list of array for each query
        queryList = []
        
        #Iterate through the rows of queryDB
        for ix in range(len(queryDB)):
            
            #Iterate through the columns of queryDB to transform data 
            for row in range(queryDB.shape[1]): 
            
                #Transform data for each column of every row
                queryList.append((queryDB.iloc[ix, row] - meanValues[row])/stdValues[row])
                
            #Append transformed list for every row  
            queryLists.append(queryList)
            
        return queryLists
        
        
'''  Mean for every column
Rating                3.754020
Revenue            2686.546078
Company Age          46.085294
Size               3231.455882
Max. Experience       4.362745
Min. Experience       2.060784
DNA                   0.217647
Cancer                0.200000
Lead                  0.560784
Biochemistry          0.403922
ML                    0.032353
Employees             0.534314
Manage                0.575490
PhD                   0.389216
Python                0.050980
Genetic               0.411765
Therapy               0.138235
No Experience         0.111765
Postdoctoral          0.061765
Immunology            0.490196
Protein               0.361765
Crystallography       0.006863
MSc                   0.300000
RNA                   0.639216
BSc                   0.400980
Vice President        0.008824
Associate             0.406863
Scientist I           0.095098
Manager               0.073529
Director              0.043137
Technician            0.057843
Analyst               0.010784
Virologist            0.002941
Investigator          0.008824
I                     0.277451
II                    0.089216
Senior                0.258824
Principal             0.031373
VP                    0.003922
Supervisor            0.010784
Researcher            0.420588
Scientist II          0.019608
Fellow                0.013725
QC                    0.024510
Scientist             0.347059
Technologist          0.092157
Engineer              0.060784
   
St. Dev for every column
Rating                0.696573
Revenue            3970.506680
Company Age          49.696089
Size               4004.214919
Max. Experience       2.393109
Min. Experience       1.534117
DNA                   0.412849
Cancer                0.400196
Lead                  0.496535
Biochemistry          0.490923
ML                    0.177022
Employees             0.499066
Manage                0.494511
PhD                   0.487812
Python                0.220066
Genetic               0.492394
Therapy               0.345316
No Experience         0.315231
Postdoctoral          0.240846
Immunology            0.500149
Protein               0.480747
Crystallography       0.082597
MSc                   0.458482
RNA                   0.480464
BSc                   0.490337
Vice President        0.093564
Associate             0.491490
Scientist I           0.293494
Manager               0.261132
Director              0.203266
Technician            0.233561
Analyst               0.103337
Virologist            0.054179
Investigator          0.093564
I                     0.447961
II                    0.285195
Senior                0.438203
Principal             0.174408
VP                    0.062530
Supervisor            0.103337
Researcher            0.493896
Scientist II          0.138716
Fellow                0.116406
QC                    0.154701
Scientist             0.476268
Technologist          0.289389
Engineer              0.239051'''