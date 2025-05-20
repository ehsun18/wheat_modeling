# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
from datetime import datetime, date

# %%
wheat_database = "/Users/hn/Documents/01_research_data/Other_people/Ehsan/wheat/"
data_dir = wheat_database + "data/"
separate_varieties_dir = data_dir + "varieties/"

reOrganized_dir = data_dir + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
# This is not what we want.
# We want full data series.
# filename = (reOrganized_dir + "all_stages_df22805_varietyAvgd.sav")
# all_stages_df22805_varietyAvgd = pd.read_pickle(filename)
# all_stages_df22805_varietyAvgd.keys()

# all_stages_df22805_varietyAvgd = all_stages_df22805_varietyAvgd["all_stages_data"]
# all_stages_df22805_varietyAvgd.head(2)

# %%
filename = (reOrganized_dir + "variables_dict.sav")
variables_dict = pd.read_pickle(filename)
variables_dict.keys()

variables_dict = variables_dict["variables_abb_dict"]
variables_dict

# %%
filename = (reOrganized_dir + "average_and_seperate_varieties_weekly.sav")
average_and_seperate_weekly = pd.read_pickle(filename)
average_and_seperate_weekly.keys()

# %%
separate_varieties_weekly = average_and_seperate_weekly["separate_varieties_weekly"]
separate_varieties_weekly.head(2)

# %%
import re

col_names = list(separate_varieties_weekly.columns)
col_names[:4]

# %%
## detect columns that start with a digit
## so we can extract them and put them in a list as time series.
pattern = r"^\d"
digital_columns = [s for s in col_names if re.match(pattern, s)]
digital_columns[:4]

# %%
print (f"{len(digital_columns) = }")
print (f"{len(col_names) = }")

# %%
non_digital_columns = [s for s in col_names if not(re.match(pattern, s))]
non_digital_columns[:4]

# %%
# count number of different variables
variables_list = [s.split("_")[1] for s in digital_columns]
variables_list = list(set(variables_list))
variables_list.remove("dtr.1")
variables_list

# %%
rows = len(separate_varieties_weekly)
cols = len(non_digital_columns) + len(variables_list)
separate_vars_weekly_TS = pd.DataFrame(np.zeros((rows, cols)))
separate_vars_weekly_TS.head(2)

# %%
columns = non_digital_columns + variables_list
separate_vars_weekly_TS.columns = columns
separate_vars_weekly_TS.head(2)

# %% [markdown]
# ## Populate dataframe

# %%
separate_vars_weekly_TS[non_digital_columns] = separate_varieties_weekly[non_digital_columns]
separate_vars_weekly_TS.head(2)

# %%
for a_variable in variables_list:
    
