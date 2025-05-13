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

# %% [markdown]
# ```00_read_excel_dump_weekly.ipynb ``` was crated on (or earlier than)  Nov. 11. 2024.
#
# Then, data were aggregated: grouped by location and year, and averaged over varieties.
#
#
# I am writing this code to take care of a file called ```df_22805.csv``` that I renamed to ```all_stages_data.csv```.

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
from datetime import datetime, date

# %%

# %%
wheat_database = "/Users/hn/Documents/01_research_data/Other_people/Ehsan/wheat/"
data_dir = wheat_database + "data/"
separate_varieties_dir = data_dir + "varieties/"

reOrganized_dir = data_dir + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
all_stages_data = pd.read_csv(data_dir + "all_stages_VarietyAvgd.csv")
all_stages_data.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
all_stages_data.head(2)

# %%
# "average_and_seperate_varieties_weekly.sav"
# average_and_seperate_varieties =  pd.read_pickle(reOrganized_dir + )
# average_and_seperate_varieties["source_code"]
# average_and_seperate_varieties['Date']

# %%
filename = (reOrganized_dir + "all_stages_df22805_varietyAvgd.sav")

export_ = {"all_stages_data": all_stages_data,
           "source_code": "00_organize_data",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          }

pickle.dump(export_, open(filename, "wb"))

# %%
all_stages_data
