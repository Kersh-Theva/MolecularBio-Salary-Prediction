#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:32:36 2020

@author: kershtheva
"""
import glassdoor_scraper as gs
import pandas as pd

path = "/Users/kershtheva/Desktop/Data Science Projects/chromedriver"

#glassdoorDB = gs.get_jobs('Molecular Biology', 400, False, path, 5)

#glassdoorDBnext400 = gs.get_jobs('Molecular Biology', 400, False, path, 5)

glassdoorDBlast400 = gs.get_jobs('Molecular Biology', 400, False, path, 5)

frames = [glassdoorDB, glassdoorDBnext400, glassdoorDBlast400]

result = pd.concat(frames)