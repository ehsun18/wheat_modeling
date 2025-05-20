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
# This file was crated on (or earlier than)  Nov. 11. 2024.
#
# Then, data are aggregated: grouped by location and year, and averaged over varieties.
# The data here are weekly, and then we have computed stuff annually and based on 4 season in a year.
#
# So, I am adding weekly to the name of this notebook.

# %%

# %%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle, pandas as pd
from datetime import datetime
import os

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/Other_people/Ehsan/wheat/"

wheat_reOrganized = data_dir_base + "wheat_reOrganized/"
wheat_plot_dir = data_dir_base + "plots/"

os.makedirs(wheat_reOrganized, exist_ok=True)
os.makedirs(wheat_plot_dir, exist_ok=True)

# %%
# # !pip3 install openpyxl

# %%
merged_varieties = pd.read_excel(data_dir_base + "merged_varieties.xlsx")
merged_with_vars = pd.read_excel(data_dir_base + "merged_with_vars.xlsx")

# %%
merged_varieties.drop(["Location", "Year"], axis="columns", inplace=True)
merged_varieties.head(2)

# %%
last_part_of_cols = [x.split("_")[-1] for x in merged_varieties.columns]
last_part_of_cols = sorted(last_part_of_cols)

# %%
last_part_of_cols = list(set(last_part_of_cols))
last_part_of_cols

# %%
variables_abb_dict = {
    "fdd" : "freezing_dd",
    "srad": "shortwave_rad_Wm2" ,
    "prdtr" : "precip_div_dtr",
    "dtr" : "diurnal_temp",
    "dgdd" : "diurnal_gdd",
    "ravg" : "relative_humidity_avg", 
    "hdd" : "high_dd",
    "vs" : "wind_speed_ms",
    "vpd" : "vpd_kPa"}

# I added the _weekly to the name on May 8, 2025
# We re working on averaged data (grouped by location and year; averaged over varieties)
filename = wheat_reOrganized + "variables_dict.sav"

export_ = {"variables_abb_dict": variables_abb_dict, 
           "source_code" : "00_read_excel_dump_weekly",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%

# %%

# %%
list(merged_varieties.columns)[:5]

# %%
print ((merged_varieties["7_dtr"] == merged_varieties["7_dtr.1"]).sum())
print ((merged_varieties["8_dtr"] == merged_varieties["8_dtr.1"]).sum())

# %% [markdown]
# ### Clean up

# %%
dtr_cols = [x for x in merged_varieties.columns if "dtr" in x]
dtr_cols_repetitions = [x for x in dtr_cols if "." in x]
print (merged_varieties.shape)
merged_varieties.drop(dtr_cols_repetitions, axis="columns", inplace=True)
print (merged_varieties.shape)

# %%
merged_varieties.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
merged_with_vars.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

# %%
list(merged_with_vars.columns)[:5]

# %% [markdown]
# #### Rename columns so I can remember?

# %%
merged_with_vars.rename(columns={"grain_yield": "yield"}, inplace=True)
merged_varieties.rename(columns={"grain_yield": "yield"}, inplace=True)

# %% [markdown]
# #### reorder columns

# %%
merged_varieties.columns

# %%
lyy = ["location", "year", "yield"]
others_ = [x for x in merged_varieties.columns if not(x in lyy)]
len(others_) + len(lyy) == len(merged_varieties.columns)

# %%
merged_varieties = merged_varieties[lyy + others_]

# %%
lyy = ["location", "year", "variety", "yield"]
others_ = [x for x in merged_with_vars.columns if not(x in lyy)]
len(others_) + len(lyy) == len(merged_with_vars.columns)

# %%
merged_with_vars = merged_with_vars[lyy + others_]

# %%
wheat_date = pd.read_csv(data_dir_base + "spring_wheat_date.csv")
wheat_date.head(2)

# %%
wheat_date.rename(columns={"planting_doy": "plant_doy",
                           "harvesting_doy": "harvest_doy"}, inplace=True)

wheat_date["season_length"] = wheat_date["harvest_doy"] - wheat_date["plant_doy"]
wheat_date.head(2)

# %%
wheat_date["heading_date"] =  pd.to_datetime(wheat_date["heading_date"])
wheat_date["harvest_date"] =  pd.to_datetime(wheat_date["harvest_date"])
wheat_date["planting_date"] =  pd.to_datetime(wheat_date["planting_date"])

# %%
print (len(wheat_date))
wheat_date['heading_date'].isna().sum()

# %%
print (wheat_date["planting_date"].dt.year.min())
print (wheat_date["planting_date"].dt.year.max())

# %%
wheat_date['season_length'].isna().sum()

# %%
pd.to_datetime(wheat_date["planting_date"])[:3]

# %%
wheat_date.head(2)

# %%
merged_with_vars.head(2)

# %%
print (len(merged_with_vars["variety"].unique()))
print (len(merged_with_vars["location"].unique()))
print ((merged_with_vars["year"].min()))
print ((merged_with_vars["year"].max()))

# %%
for a_loc in merged_with_vars["location"].unique():
    df = merged_with_vars[merged_with_vars["location"] == a_loc]
    if len(df["variety"].unique()) != 13:
        print (a_loc, len(df["variety"].unique()))

# %%
varieties  = list(merged_with_vars["variety"].unique())
Plaza_variety  = list(merged_with_vars[merged_with_vars["location"] == "Plaza"]["variety"].unique())

[x for x in varieties if not (x in Plaza_variety)]

# %%
wheat_date.head(2)

# %%
merged_with_vars.head(2)

# %%
tick_legend_FontSize = 14
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5,
          'legend.handlelength': 2,
          'axes.grid' : False}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

suptitle_fontsize = 15
dpi_ = 300

# %%
df = wheat_date.copy()
df.sort_values(by="season_length", inplace=True)
df.reset_index(inplace=True, drop=True)
df.head(2)

min_loc = df.loc[0, "location"]
min_year = df.loc[0, "year"]

two_max_idx = list(df.index)[-2:]
max_loc = df.loc[two_max_idx[1], "location"]
max_year = df.loc[two_max_idx[1], "year"]

max2_loc = df.loc[two_max_idx[0], "location"]
max2_year = df.loc[two_max_idx[0], "year"]

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})

sns.histplot(data=df["season_length"], ax=axes, bins=100, kde=True);
# axes.legend(["ANPP (mean lb/acr)"], loc='upper right');
axes.set_xlabel("season length (in days)");

text_ = min_loc + ", " + str(min_year)
axes.text(df.loc[0, "season_length"]-2, 3, text_, fontsize = 12);

text_ = max_loc + ", " + str(max_year) + "\n " + max2_loc + ", " + str(max2_year) 
axes.text(df.loc[two_max_idx[0], "season_length"]-15, 3, text_, fontsize = 12);

# axes.set_title('season length distribution');
fig.suptitle('season length distribution', y=0.95, fontsize=suptitle_fontsize)
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.052, right=0.981, wspace=-0.2, hspace=0)
file_name = wheat_plot_dir + "season_length_hist.pdf"
plt.savefig(file_name, dpi=400)

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})

sns.histplot(data=wheat_date["plant_doy"], ax=axes, bins=100, kde=True);
# axes.legend(["ANPP (mean lb/acr)"], loc='upper right');
axes.set_xlabel("DoY");

# axes.set_title('season length distribution');
fig.suptitle('planting date distribution', y=0.95, fontsize=suptitle_fontsize)
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.052, right=0.981, wspace=-0.2, hspace=0)
file_name = wheat_plot_dir + "planting_DoY_hist.pdf"
plt.savefig(file_name, dpi=400)

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})

sns.histplot(data=merged_with_vars["yield"], ax=axes, bins=100, kde=True);
axes.set_xlabel("yield (bushel/acre)");

# axes.set_title('season length distribution');
fig.suptitle('yield distribution (bushel/acre)', y=0.95, fontsize=suptitle_fontsize)
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.052, right=0.981, wspace=-0.2, hspace=0)
file_name = wheat_plot_dir + "yield_hist.pdf"
plt.savefig(file_name, dpi=400)

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})

sns.histplot(data=np.log(merged_with_vars["yield"]), ax=axes, bins=100, kde=True);
axes.set_xlabel("log(yield) (yield: bushel/acre)");
fig.suptitle('log(yield) distribution (yield: bushel/acre)', y=0.95, fontsize=suptitle_fontsize)
# axes.set_title('season length distribution');
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.052, right=0.981, wspace=-0.2, hspace=0)
file_name = wheat_plot_dir + "log_yield_hist.pdf"
plt.savefig(file_name, dpi=400)

# %%
temp_df = wheat_date.copy()
temp_df.dropna(inplace=True)
print (temp_df.shape)
temp_df.head(2)

# %%
temp_df["heading_date"] = pd.to_datetime(temp_df["heading_date"])
temp_df["heading_DoY"] = temp_df["heading_date"].dt.dayofyear
temp_df.head(2)

# %%
temp_df["Head_minus_plant"] = temp_df["heading_DoY"] - temp_df["plant_doy"]

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})

sns.histplot(data=(temp_df["Head_minus_plant"]), ax=axes, bins=100, kde=True);
axes.set_xlabel(r"(heading - planting) dates");
fig.suptitle('planting date to heading date', y=0.95, fontsize=suptitle_fontsize)
# axes.set_title('heading_date');
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.052, right=0.981, wspace=-0.2, hspace=0)
file_name = wheat_plot_dir + "heading_date_dist.pdf"
plt.savefig(file_name, dpi=400)

# %%
print (wheat_date.shape)
print (temp_df.shape)

# %%
229-113

# %% [markdown]
# ### annual DF, Seperate Varieties

# %%
# grab dgdd and precip
# drop dgdd
gdd_cols = [x for x in merged_with_vars.columns if ("gdd" in x)]
gdd_cols = [x for x in gdd_cols if not ("dgdd" in x)]

dgdd_cols = [x for x in merged_with_vars.columns if ("dgdd" in x)]
precip_cols = [x for x in merged_with_vars.columns if ("precip" in x)]

x_vars = gdd_cols + dgdd_cols + precip_cols

# %%
wheat_date.head(2)

# %%
wanted_cols = ['location', 'year', 'variety', 'yield'] + x_vars
df = merged_with_vars[wanted_cols].copy()
df = pd.merge(df, wheat_date[["year", "location", "season_length"]], on=["year", "location"], how="left")

df.fillna(value=0, inplace=True)
df.head(2)

# %%
dict_season = {"location" : list(df["location"]),
               "year" : list(df["year"]),
               "variety" : list(df["variety"]),
               "yield" : list(df["yield"]),
              }
df_year = pd.DataFrame(dict_season)
df_year.head(2)

# %%
df_year["year_gdd"] = df[gdd_cols].sum(axis=1)
df_year["year_dgdd"] = df[dgdd_cols].sum(axis=1)
df_year["year_precip"] = df[precip_cols].sum(axis=1)
df_year.head(2)

# %%

# %% [markdown]
# ### 4 Season DF, Seperate Varieties

# %%
subseason_count = 4
df["season_length_weeks"] = df["season_length"] / 7
df["week_count_per_season"] = round(df["season_length_weeks"] / subseason_count)
df["week_count_per_season"] = df["week_count_per_season"].astype(int)
df.head(2)

# %%
dict_season = {"location" : list(df["location"]),
               "year" : list(df["year"]),
               "variety" : list(df["variety"]),
               "yield" : list(df["yield"]),
              }
df_season = pd.DataFrame(dict_season)
df_season.head(2)

# %%
desired_features = ["gdd", "dgdd", "precip"]
# form new season-wise columns
pre_var = ["s" + str(x) + "_" for x in list(range(1, subseason_count+1))]
pre_var = pre_var * len(desired_features)
post_feature = sorted(desired_features * 4)
season_cols = [x + y for x, y in zip(pre_var, post_feature)]

# add new columns to df_season
df_season[season_cols] = -10
df_season.head(2)

# %%
# %%time

# we have to do a for-loop
for idx in df_season.index:
    curr_df = df.loc[idx].copy()
    week_count_per_season = curr_df["week_count_per_season"]
    # print (curr_df["location"] + " - " + str(curr_df["year"]))
    
    # The cuts between each season
    cut_offs = list(week_count_per_season * range(1, subseason_count+1))
    # since there is only 25 weeks worth of data
    # replace the last cutoff to 25 so that last season
    # contains everything.
    cut_offs[-1] = 25
    # takes care of first season
    cut_offs = [0] + cut_offs
    
    for season_ in range(1, subseason_count+1):
        start_week = cut_offs[season_-1] + 1
        end_week   = cut_offs[season_] + 1
        for var_ in desired_features:
            weekly_cols = [str(x) + "_" + var_ for x in range(start_week, end_week)]
            df_season.loc[idx, "s"+str(season_)+"_"+var_] = curr_df[weekly_cols].sum()

df_season.head(2)

# %%
df_season[(df_season["location"] == "Almira") & (df_season["variety"] == "Alpowa")]

# %% [markdown]
# ## Annual X, Average Varieties

# %%
df_year.head(2)

# %%
df_year_avg = df_year[["location", "year", "yield"]].copy()
df_year_avg = df_year_avg.groupby(["location", "year"]).mean().reset_index(drop=False)

df_year_weather = df_year[["location", "year", "year_gdd", "year_dgdd", "year_precip"]].copy()
df_year_weather.drop_duplicates(inplace=True)

df_year_avg = pd.merge(df_year_avg, df_year_weather, on=["location", "year"], how="left")
df_year_avg.head(2)

# %% [markdown]
# ## 4 Seasons X, Average Varieties

# %%
df_season.head(2)

# %%
seasonal_vars = [x for x in df_season.columns if "s" in x]

# %%
df_season_avg = df_season[["location", "year", "yield"]].copy()
df_season_avg = df_season_avg.groupby(["location", "year"]).mean().reset_index(drop=False)

df_season_weather = df_season[["location", "year"] + seasonal_vars].copy()
df_season_weather.drop_duplicates(inplace=True)

df_season_avg = pd.merge(df_season_avg, df_season_weather, on=["location", "year"], how="left")
df_season_avg.head(2)

# %%
print (df_season.shape)
print (df_season_avg.shape)

# %%
merged_varieties["yield"] = merged_varieties["yield"].round(2)
merged_with_vars["yield"] = merged_with_vars["yield"].round(2)

df_year["yield"] = df_year["yield"].round(2)
df_year_avg["yield"] = df_year_avg["yield"].round(2)

df_season["yield"] = df_season["yield"].round(2)
df_season_avg["yield"] = df_season_avg["yield"].round(2)

# %%

# %%
# I added the _weekly to the name on May 8, 2025
# We re working on averaged data (grouped by location and year; averaged over varieties)
filename = wheat_reOrganized + "average_and_seperate_varieties_weekly.sav"

export_ = {"averaged_varieties_weekly": merged_varieties, 
           "separate_varieties_weekly": merged_with_vars,

           "separate_varieties_annual": df_year, 
           "averaged_varieties_annual": df_year_avg, 
           
           "separate_varieties_4season": df_season,
           "averaged_varieties_4season" : df_season_avg,
           
           "dates" : wheat_date,
           "source_code" : "read_excel_dump",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%

# %%
