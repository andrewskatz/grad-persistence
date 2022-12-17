# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:31:58 2022

@author: akatz4
"""



from personal_utilities import zs_labeling as zsl

# import transformers


# from transformers import pipeline


# from personal_utilities import embed_cluster as ec


import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


import pickle


"""

Import data and preprocess

"""


topic = "grad_persist"



os.getcwd()

# for jee df with abstracts and full article information
proj_path = 'C:\\Users\\akatz4\\OneDrive - Virginia Tech\\Desktop\\cb psu nlp'

os.chdir(proj_path)
os.listdir()


raw_df = pd.read_csv("AttritionSMS_OpenEnd_Coded.csv")


id_vars = []
value_vars = []





all_vars = raw_df.columns
value_vars = [var for var in all_vars if "Q8_1_" in var and "Codes" not in var]
print(value_vars)
id_vars = [var for var in all_vars if var not in value_vars]
print(id_vars)


long_df = pd.melt(raw_df, id_vars=id_vars, value_vars=value_vars, var_name = 'response_date', value_name = 'text_col')



# =============================================================================
# Load labels
# =============================================================================

labels_df = pd.read_csv("grad_persist_codebook.csv")
print(labels_df.columns)

level_one_labels = list(labels_df['first_level'].unique())
print(len(level_one_labels))

level_two_labels = list(labels_df['second_level'].unique())
print(len(level_two_labels))



"""

Labeling

"""


# classifier = pipeline(task = 'zero-shot-classification', model = 'facebook/bart-large-mnli')



long_df.dropna(subset=['text_col'])

filtered_df = long_df.dropna(subset=['text_col'])
filtered_df['new_sent_id'] = filtered_df.index


zs_threshold = 0.12


test_df = filtered_df.head(5)
print(test_df.columns)


# =============================================================================
# using the utility file zs_labeling.py and label_df_with_zs()
# =============================================================================


# test_results_df = zs.label_df_with_zs(test_df, 'text_col', 'Participant', level_one_labels, 0.1)

test_df = filtered_df.head(100)

zs_threshold = 0.6

text_col_name = 'text_col'
id_col_name = 'Participant'
multi_label = True

total_results_df = zsl.label_df_with_zs(test_df, 
                                        text_col_name, 
                                        id_col_name, 
                                        level_two_labels, 
                                        zs_threshold, 
                                        multi_label=multi_label)


save_date = "20221211"
zs_threshold_save = str(zs_threshold).replace('.', '-')
if multi_label == True:
    total_results_df.to_csv(f"{topic}_zs_label_{zs_threshold_save}thresh_multi_{save_date}.csv", index = False)

if multi_label == False:
    total_results_df.to_csv(f"{topic}_zs_label_{zs_threshold_save}thresh_no-multi_{save_date}.csv", index = False)









# =============================================================================
# old school way (without the utility file and label_df_with_zs)
# =============================================================================
total_results_df = pd.DataFrame(columns=['sequence', 'labels', 'scores', 'participant', 'new_sent_id'])




for index, row in test_df.iterrows():
  row_text = row['text_col']
  new_sent_id = row['new_sent_id']
  participant = row['Participant']
  
  print(f"working on item {index} with id {new_sent_id}: {row_text}")
  
  classifier_results = classifier(row_text, level_two_labels)
  results_df = pd.DataFrame(classifier_results)
  results_df['new_sent_id'] = new_sent_id
  results_df['participant'] = participant
  
  results_df = results_df[results_df['scores'] > zs_threshold]

  total_results_df = pd.concat([total_results_df, results_df])










