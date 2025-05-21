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


import pandas as pd
import numpy as np
import os, os.path, pickle, sys
from datetime import datetime, date

from sklearn.model_selection import train_test_split

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

# %% [markdown]
# ### Drop columns with name dtr.1 in them from dataframe.

# %%
print (separate_varieties_weekly.shape)
dt1_cols = [s for s in separate_varieties_weekly.columns if "dtr.1" in s]
separate_varieties_weekly.drop(columns=dt1_cols, inplace=True)
print (separate_varieties_weekly.shape)

# %%
## Get rid of location, year, variety and turn them into ID
separate_varieties_weekly["ID"] = separate_varieties_weekly["location"] + "_" +\
                                  separate_varieties_weekly["year"].astype(str) + "_" +\
                                  separate_varieties_weekly["variety"]


separate_varieties_weekly.drop(columns=["location", "year", "variety"], inplace=True)
separate_varieties_weekly.head(2)

# %%
# Convert the yield to string to save the data in .TS format

separate_varieties_weekly["yield"] = separate_varieties_weekly["yield"].astype("str")

# %%
import re

col_names = list(separate_varieties_weekly.columns)
col_names[:4]

# %%
## detect columns that start with a digit
## so we can extract them and put them in a list as time series.
## re.match() only matches at the beginning of the string.
## re.search() looks for a match anywhere in the string:
pattern = r"^\d"
digital_columns = [s for s in col_names if re.match(pattern, s)]
digital_columns[:4]

# %%
print (f"{len(digital_columns) = }")
print (f"{len(col_names) = }")

# %%
non_digital_columns = sorted([s for s in col_names if not(re.match(pattern, s))])
non_digital_columns[:4]

# %%
# count number of different variables
variables_list = [s.split("_")[1] for s in digital_columns]
variables_list = list(set(variables_list))
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

# %%

# %%

# %% [markdown]
# ## Populate dataframe

# %%
separate_vars_weekly_TS[non_digital_columns] = separate_varieties_weekly[non_digital_columns]

separate_vars_weekly_TS_NaNZeros = separate_vars_weekly_TS.copy()
separate_vars_weekly_TS_NaNRand = separate_vars_weekly_TS.copy()
separate_vars_weekly_TS.head(2)

# %%
separate_varieties_weekly.head(2)

# %%
# week_cnt = np.array(sorted(list(set([int(s.split("_")[0]) for s in digital_columns]))))
# max_number_of_weeks = week_cnt.max()
# max_number_of_weeks

# %%
## create a dictionary with keys that are name of variables
## and its values are list of weekly columns associated with that variable
variables_colums_grp = dict()
for key_ in variables_list:
    variables_colums_grp[key_] = [s for s in col_names if re.search(key_, s)]

# %%
## Populate the dataframe with list of time-series in each cell
for a_variable in variables_list:
    curr_columns = variables_colums_grp[a_variable]
    ts = separate_varieties_weekly[curr_columns].values.tolist()
    
    # convert each of them to a pd.Series
    # to be used lated for export .ts files
    ts = [pd.Series(s) for s in ts]
    
    # convert it to pandas series?
    separate_vars_weekly_TS[a_variable] = ts

# %%

# %% [markdown]
# #### Replace NaNs 
# with zeros and random variables, possibly, to be used in mv-ts-transformers.

# %%
separate_varieties_weekly_NaNZeros = separate_varieties_weekly.copy()
separate_varieties_weekly_NaNRand = separate_varieties_weekly.copy()

# %%
separate_varieties_weekly_NaNZeros = separate_varieties_weekly_NaNZeros.fillna(0)

import random
random.seed(42)
np.random.seed(42)
for col in separate_varieties_weekly_NaNRand.columns:
    na_mask = separate_varieties_weekly_NaNRand[col].isna()
    # random integers between 0 and 9
    separate_varieties_weekly_NaNRand.loc[na_mask, col] = np.random.normal(size=na_mask.sum())

separate_varieties_weekly_NaNRand.head(2)

# %%
## Populate the dataframe with list of time-series in each cell
for a_variable in variables_list:
    curr_columns = variables_colums_grp[a_variable]
    ts = separate_varieties_weekly_NaNZeros[curr_columns].values.tolist()
    
    # convert each of them to a pd.Series
    # to be used lated for export .ts files
    ts = [pd.Series(s) for s in ts]
    
    # convert it to pandas series?
    separate_vars_weekly_TS_NaNZeros[a_variable] = ts

# %%
## Populate the dataframe with list of time-series in each cell
for a_variable in variables_list:
    curr_columns = variables_colums_grp[a_variable]
    ts = separate_varieties_weekly_NaNRand[curr_columns].values.tolist()
    
    # convert each of them to a pd.Series. 
    # to be used lated for export .ts files
    ts = [pd.Series(s) for s in ts]
    
    # convert it to pandas series?
    separate_vars_weekly_TS_NaNRand[a_variable] = ts

# %%
separate_vars_weekly_TS_NaNRand.head(2)

# %%
separate_vars_weekly_TS_NaNZeros.head(2)

# %%
separate_vars_weekly_TS.head(2)

# %%
separate_varieties_weekly.head(2)

# %%
separate_vars_weekly_TS["vs"][0]

# %%
separate_vars_weekly_TS_NaNZeros["vs"][0]

# %%
separate_vars_weekly_TS_NaNRand["vs"][0]

# %% [markdown]
# ### Split 80-20 for all three datasets

# %%
X_train, X_test, y_train, y_test = train_test_split(separate_vars_weekly_TS.drop(columns=["yield"], inplace=False),
                                                    separate_vars_weekly_TS["yield"], 
                                                    test_size=0.2, random_state=42)

# %%
train_idx = list(X_train.index)
test_idx = list(X_test.index)

# %%
separate_vars_weekly_TS_train = separate_vars_weekly_TS[separate_vars_weekly_TS.index.isin(train_idx)].copy()
separate_vars_weekly_TS_test = separate_vars_weekly_TS[separate_vars_weekly_TS.index.isin(test_idx)].copy()

print (f"{separate_vars_weekly_TS_train.shape = }")
print (f"{separate_vars_weekly_TS_test.shape = }")

# %%
separate_vars_weekly_TS_NaNZeros_train = separate_vars_weekly_TS_NaNZeros[
                                            separate_vars_weekly_TS_NaNZeros.index.isin(train_idx)].copy()
separate_vars_weekly_TS_NaNZeros_test = separate_vars_weekly_TS_NaNZeros[
                                            separate_vars_weekly_TS_NaNZeros.index.isin(test_idx)].copy()

print (f"{separate_vars_weekly_TS_NaNZeros_train.shape = }")
print (f"{separate_vars_weekly_TS_NaNZeros_test.shape = }")

# %%
separate_vars_weekly_TS_NaNRand_train = separate_vars_weekly_TS_NaNRand[
                                            separate_vars_weekly_TS_NaNRand.index.isin(train_idx)].copy()
separate_vars_weekly_TS_NaNRand_test = separate_vars_weekly_TS_NaNRand[
                                            separate_vars_weekly_TS_NaNRand.index.isin(test_idx)].copy()
print (f"{separate_vars_weekly_TS_NaNRand_train.shape = }")
print (f"{separate_vars_weekly_TS_NaNRand_test.shape = }")

# %%
import sktime
from sktime.datasets import load_arrow_head
from sktime.datasets import write_dataframe_to_tsfile


# %%
# out_dir_ = reOrganized_dir + "wheat_regression_data_mvts/NaNRand/"
# os.makedirs(out_dir_, exist_ok=True)

# with open(out_dir_ + 'separate_vars_weekly_TS_NaNRand_TRAIN.ts', "w", encoding="utf-8") as f:
#     f.write(separate_vars_weekly_TS_NaNRand_train)

# %%

# %%
def ensure_series(cell):
    # If already a Series, return as-is
    if isinstance(cell, pd.Series):
        return cell
    # If list/array, wrap as Series
    elif isinstance(cell, (list, tuple)) or hasattr(cell, "__len__"):
        return pd.Series(cell)
    # Otherwise (scalar or NaN), wrap as 1-element Series
    else:
        return pd.Series([cell])


# %%
for col in X.columns:
    print (type(X[col]))

# %%
# from sktime.datatypes import check_is_scitype

# check_is_scitype(X, scitype="Panel", return_metadata=True)

# %%
for ii in list(separate_vars_weekly_TS_NaNZeros_train["yield"]):
    if type(ii)==None:
        print (ii)


# %%
def check_non_series_cells(df):
    for row_idx in df.index:
        for col_name in df.columns:
            value = df.at[row_idx, col_name]
            if not isinstance(value, pd.Series):
                print(f"Non-Series found at row {row_idx}, column '{col_name}': type={type(value)}")
                
X = separate_vars_weekly_TS_NaNZeros_train.drop(columns=["ID", "yield"])
check_non_series_cells(X)

# %%
out_dir_ = reOrganized_dir + "wheat_regression_data_mvts/separate_vars_weekly_TS_NaNZeros/"
os.makedirs(out_dir_, exist_ok=True)

X = separate_vars_weekly_TS_NaNZeros_train.drop(columns=["ID", "yield"]).copy()
X.reset_index(drop=True, inplace=True)
y = pd.Series(separate_vars_weekly_TS_NaNZeros_train["yield"])

write_dataframe_to_tsfile(data=X, 
                          class_label = y, 
                          class_value_list=y.tolist(),
                          path=out_dir_,
                          problem_name= "separate_vars_weekly_TS_NaNZeros_TRAIN")


del(X,y)

X = separate_vars_weekly_TS_NaNZeros_test.drop(columns=["ID", "yield"]).copy()
X.reset_index(drop=True, inplace=True)
y = pd.Series(separate_vars_weekly_TS_NaNZeros_test["yield"])

write_dataframe_to_tsfile(data=X, 
                          class_label = y, 
                          class_value_list=y.tolist(),
                          path=out_dir_,
                          problem_name= "separate_vars_weekly_TS_NaNZeros_TEST")

# %%
out_dir_ = reOrganized_dir + "wheat_regression_data_mvts/separate_vars_weekly_TS_NaNRand/"
os.makedirs(out_dir_, exist_ok=True)

X = separate_vars_weekly_TS_NaNRand_train.drop(columns=["ID", "yield"]).copy()
X.reset_index(drop=True, inplace=True)
y = pd.Series(separate_vars_weekly_TS_NaNRand_train["yield"])

write_dataframe_to_tsfile(data=X, 
                          class_label = y, 
                          class_value_list=y.tolist(),
                          path=out_dir_,
                          problem_name= "separate_vars_weekly_TS_NaNRand_TRAIN")


del(X,y)

X = separate_vars_weekly_TS_NaNRand_test.drop(columns=["ID", "yield"]).copy()
X.reset_index(drop=True, inplace=True)
y = pd.Series(separate_vars_weekly_TS_NaNRand_test["yield"])

write_dataframe_to_tsfile(data=X, 
                          class_label = y, 
                          class_value_list=y.tolist(),
                          path=out_dir_,
                          problem_name= "separate_vars_weekly_TS_NaNRand_TEST")

# %%
out_dir_ = reOrganized_dir + "wheat_regression_data_mvts/separate_vars_weekly_TS_wNaN/"
os.makedirs(out_dir_, exist_ok=True)

X = separate_vars_weekly_TS_train.drop(columns=["ID", "yield"]).copy()
X.reset_index(drop=True, inplace=True)
y = pd.Series(separate_vars_weekly_TS_train["yield"])

write_dataframe_to_tsfile(data=X, 
                          class_label = y, 
                          class_value_list=y.tolist(),
                          path=out_dir_,
                          problem_name= "separate_vars_weekly_TS_wNaN_TRAIN")


del(X,y)

X = separate_vars_weekly_TS_test.drop(columns=["ID", "yield"]).copy()
X.reset_index(drop=True, inplace=True)
y = pd.Series(separate_vars_weekly_TS_test["yield"])

write_dataframe_to_tsfile(data=X, 
                          class_label = y, 
                          class_value_list=y.tolist(),
                          path=out_dir_,
                          problem_name= "separate_vars_weekly_TS_wNaN_TEST")


# %%
def compare_dataframes_by_position(df1, df2):
    if df1.shape != df2.shape:
        print(f"DataFrames have different shapes: {df1.shape} vs {df2.shape}")
        return

    unequal = False
    for i in range(df1.shape[0]):  # Iterate rows by position
        for j in range(df1.shape[1]):  # Iterate columns by position
            val1 = df1.iat[i, j]
            val2 = df2.iat[i, j]
            if isinstance(val1, pd.Series) and isinstance(val2, pd.Series):
                if not val1.equals(val2):
                    print(f"Difference at row={i}, col={j}:")
                    print(f"  df1: {val1}")
                    print(f"  df2: {val2}")
                    unequal = True
            else:
                if val1 != val2:
                    print(f"Difference at row={i}, col={j}: df1={val1}, df2={val2}")
                    unequal = True

    if not unequal:
        print("The DataFrames are identical by content (column names ignored).")



# %%

# %%
# Check if it worked:
dir_ = reOrganized_dir + "wheat_regression_data_mvts/separate_vars_weekly_TS_NaNRand/"
f_name = "separate_vars_weekly_TS_NaNRand_TRAIN.ts"
A, A_labels = sktime.datasets.load_from_tsfile_to_dataframe(dir_ + f_name,
                                                            return_separate_X_and_y=True, 
                                                            replace_missing_vals_with='NaN')

df2 = separate_vars_weekly_TS_NaNRand_train.drop(columns=["ID", "yield"]).copy()
compare_dataframes_by_position(A, df2)

print ((list(separate_vars_weekly_TS_NaNRand_train["yield"]) == A_labels).sum() == len(A_labels))

print ("----------   Testset.   ----------")

f_name = "separate_vars_weekly_TS_NaNRand_TEST.ts"
A, A_labels = sktime.datasets.load_from_tsfile_to_dataframe(dir_ + f_name,
                                                            return_separate_X_and_y=True, 
                                                            replace_missing_vals_with='NaN')

df2 = separate_vars_weekly_TS_NaNRand_test.drop(columns=["ID", "yield"]).copy()
compare_dataframes_by_position(A, df2)

print ((list(separate_vars_weekly_TS_NaNRand_test["yield"]) == A_labels).sum() == len(A_labels))

# %%
# Check if it worked:
dir_ = reOrganized_dir + "wheat_regression_data_mvts/separate_vars_weekly_TS_NaNZeros/"
f_name = "separate_vars_weekly_TS_NaNZeros_TRAIN.ts"

A, A_labels = sktime.datasets.load_from_tsfile_to_dataframe(dir_ + f_name,
                                                            return_separate_X_and_y=True, 
                                                            replace_missing_vals_with='NaN')

df2 = separate_vars_weekly_TS_NaNZeros_train.drop(columns=["ID", "yield"]).copy()
compare_dataframes_by_position(A, df2)

print ((list(separate_vars_weekly_TS_NaNZeros_train["yield"]) == A_labels).sum() == len(A_labels))

print ("----------   Testset.   ----------")

f_name = "separate_vars_weekly_TS_NaNZeros_TEST.ts"
A, A_labels = sktime.datasets.load_from_tsfile_to_dataframe(dir_ + f_name,
                                                            return_separate_X_and_y=True, 
                                                            replace_missing_vals_with='NaN')

df2 = separate_vars_weekly_TS_NaNZeros_test.drop(columns=["ID", "yield"]).copy()
compare_dataframes_by_position(A, df2)

print ((list(separate_vars_weekly_TS_NaNZeros_test["yield"]) == A_labels).sum() == len(A_labels))


# %%
# Check if it worked:
dir_ = reOrganized_dir + "wheat_regression_data_mvts/separate_vars_weekly_TS_wNaN/"
f_name = "separate_vars_weekly_TS_wNaN_TRAIN.ts"

A, A_labels = sktime.datasets.load_from_tsfile_to_dataframe(dir_ + f_name,
                                                            return_separate_X_and_y=True, 
                                                            replace_missing_vals_with='NaN')

df2 = separate_vars_weekly_TS_train.drop(columns=["ID", "yield"]).copy()
compare_dataframes_by_position(A, df2)

print ((list(separate_vars_weekly_TS_train["yield"]) == A_labels).sum() == len(A_labels))

print ("----------   Testset.   ----------")
f_name = "separate_vars_weekly_TS_wNaN_TEST.ts"
A, A_labels = sktime.datasets.load_from_tsfile_to_dataframe(dir_ + f_name,
                                                            return_separate_X_and_y=True, 
                                                            replace_missing_vals_with='NaN')

df2 = separate_vars_weekly_TS_test.drop(columns=["ID", "yield"]).copy()
compare_dataframes_by_position(A, df2)

print ((list(separate_vars_weekly_TS_test["yield"]) == A_labels).sum() == len(A_labels))
