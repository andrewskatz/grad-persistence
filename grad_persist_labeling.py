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
proj_path = 'C:\\Users\\akatz4\\Documents\\cb psu nlp'

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

# labels_df = pd.read_csv("grad_persist_codebook.csv")
labels_df = pd.read_csv("grad_persist_codebook_v2.csv")
print(labels_df.columns)

level_one_labels = list(labels_df['level_one_label'].unique())
print(len(level_one_labels))

level_two_labels = list(labels_df['level_two_label'].unique())
print(len(level_two_labels))



"""

Labeling

"""


# classifier = pipeline(task = 'zero-shot-classification', model = 'facebook/bart-large-mnli')



long_df.dropna(subset=['text_col'])

filtered_df = long_df.dropna(subset=['text_col'])
filtered_df['new_sent_id'] = filtered_df.index



# test_df = filtered_df.head(5)
# print(test_df.columns)


# =============================================================================
# using the utility file zs_labeling.py and label_df_with_zs()
# =============================================================================


# test_results_df = zs.label_df_with_zs(test_df, 'text_col', 'Participant', level_one_labels, 0.1)

test_df = filtered_df.head(100)

zs_threshold = 0.2

text_col_name = 'text_col'
id_col_name = 'Participant'
multi_label = False
keep_top_n = False
top_n=5

total_results_df = zsl.label_df_with_zs(test_df, 
                                        text_col_name, 
                                        id_col_name, 
                                        level_one_labels, 
                                        zs_threshold, 
                                        multi_label=multi_label,
                                        keep_top_n=keep_top_n,
                                        top_n=top_n)


save_date = "20221211"
zs_threshold_save = str(zs_threshold).replace('.', '-')
if multi_label == True:
    total_results_df.to_csv(f"{topic}_zs_label_{zs_threshold_save}thresh_multi_{save_date}.csv", index = False)

if multi_label == False:
    total_results_df.to_csv(f"{topic}_zs_label_{zs_threshold_save}thresh_no-multi_{save_date}.csv", index = False)




# =============================================================================
# second round (using level_one_labels as starting point)
# =============================================================================


t_df = total_results_df.rename(columns={'labels': 'level_one_label',
                                 'scores': 'level_one_score'})

# t_df = t_df.head(60)

level_one_used = list(t_df['level_one_label'].unique())

second_round_df = pd.DataFrame(columns=['sequence', 'labels', 
                                        'level_one_score', 'level_two_score', 
                                        'original_id', 'level_one_label', 'level_two_label'])



text_col = "sequence"
id_col = "original_id"


for level_one_label in level_one_used:
    temp_df = t_df[t_df['level_one_label'] == level_one_label]
    # print(temp_df.shape)
    level_two_to_use = labels_df[labels_df['first_level'] == level_one_label]['second_level']
    level_two_to_use = list(level_two_to_use)
    # print(level_two_to_use)
    
    
    for index, row in temp_df.iterrows():
          row_text = row[text_col]
          original_sent_id = row[id_col]
          print(f"working on item {index} with id {original_sent_id}: {row_text}")
          
          classifier_results = zsl.classifier(row_text, level_two_to_use)
          all_results_df = pd.DataFrame(classifier_results)
          all_results_df['original_id'] = original_sent_id
          all_results_df['level_one_label'] = level_one_label
          all_results_df['level_one_score'] = row['level_one_score']
          
        
        
          final_results_df = all_results_df.head(1) # start by picking the top match to make sure at least one match returned
          cutoff_results_df = all_results_df[all_results_df['scores'] > zs_threshold] # now select all results above a threshold cutoff
        
          final_results_df = pd.concat([final_results_df, cutoff_results_df]) # combine the two together
          final_results_df = final_results_df.drop_duplicates() # drop the duplicated top entry
          
          final_results_df = final_results_df.rename(columns={'labels': 'level_two_label', 'scores': 'level_two_score'})
        
          second_round_df = pd.concat([second_round_df, final_results_df])
    


second_round_df.to_csv("test_second_round_labeling_20221217.csv", index=False)




# =============================================================================
# using new second_round_zsl() function
# =============================================================================

t2_df, level_one_used = zsl.prepare_second_round(total_results_df)


r2_test_df = t2_df.head(20)
text_col = "sequence"
id_col = "original_id"
zs_threshold = 0.05

r2_df, used = zsl.second_round_zsl(r2_test_df, 
                             labels_df, 
                             text_col, 
                             id_col, 
                             zs_threshold, 
                             multi_label=multi_label, 
                             keep_top_n=keep_top_n, 
                             top_n=top_n)





# =============================================================================
# also trying direct level two labeling
# =============================================================================

t_df_2 = zsl.label_df_with_zs(t_df, 
                              text_col, 
                              id_col, 
                              level_two_labels, 
                              zs_threshold, 
                              multi_label=multi_label,
                              keep_top_n=keep_top_n,
                              top_n=top_n)



# =============================================================================
# old script
# =============================================================================


test_results_df = pd.DataFrame(columns=['sequence', 'labels', 'scores', 'original_id'])
        
        for index, row in unlabeled_df.iterrows():
          row_text = row[text_col]
          original_sent_id = row[id_col]
          print(f"working on item {index} with id {original_sent_id}: {row_text}")
          
          classifier_results = classifier(row_text, class_labels)
          all_results_df = pd.DataFrame(classifier_results)
          all_results_df['original_id'] = original_sent_id
        
        
          final_results_df = all_results_df.head(1) # start by picking the top match to make sure at least one match returned
          cutoff_results_df = all_results_df[all_results_df['scores'] > zs_threshold] # now select all results above a threshold cutoff
        
          final_results_df = pd.concat([final_results_df, cutoff_results_df]) # combine the two together
          final_results_df = final_results_df.drop_duplicates() # drop the duplicated top entry
        
          test_results_df = pd.concat([test_results_df, final_results_df])
      






"""
old material

"""




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










