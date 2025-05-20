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

variables_dict = variables_dict["variables_abb_dict"]
variables_dict

# %%
