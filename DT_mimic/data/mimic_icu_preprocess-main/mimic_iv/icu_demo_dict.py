# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:06:13 2025

@author: liyanke
"""
import os
import pickle
import pandas as pd
from pathlib import Path


demo_folder = Path("processed_icu")

DICT_DIR = Path(os.getcwd()) / "dict"
DICT_DIR.mkdir(exist_ok=True)  

# read the icu_demo summary file
icu_demo = pd.read_csv(demo_folder / "icu_demo.csv")

# List of columns to encode
categorical_cols = [
    'icu_type', 
    'gender', 
    'admission_type', 
    'insurance', 
    'language', 
    'marital_status', 
    'race'
]


# Generate numerical encoding for each categorical column
for col in categorical_cols:
    condVocabDict={}
    
    if col == 'gender':
        condVocabDict['<PAD>']=0
        condVocabDict['M']=1
        condVocabDict['F']=2    
    else:
        condVocabDict[0]=0
        # Get unique sorted values (dropping any missing values)
        unique_vals = sorted(icu_demo[col].dropna().unique())
        for val in range(len(unique_vals)):
            condVocabDict[unique_vals[val]]= val+1
    
    # Save the encoding dictionaries as a pickle file.
    with open(DICT_DIR / f"{col}Dict", "wb") as f:
        pickle.dump(condVocabDict, f)
        