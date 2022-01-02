#!/usr/bin/env python
# coding: utf-8

# # GVB predictions - 1 week ahead - Stop level (metro and tram/bus)

# ## To-do:

# * env file needs to be loaded. Note that "from sqlalchemy import create_engine" and "import env" needs to be uncommented in the third cell.
# 
# * Predictions need to be stored in the database.
# 
# * Decide where to calculate crowd levels. Do this based on static csv from GVB.
# 
# * Events are specified manually and are taken from a static file at this moment. As a temporary solution, we can save events in the database. Ideally, we need to get events from an API.
# 
# * Historical weather data and weather forecast have different units for some columns? This is currently solved by multiplying everything by 10, which should be correct.

# ## Preparations

# In[ ]:


get_ipython().run_cell_magic('capture', '', "get_ipython().run_cell_magic('bash', '', 'pip install psycopg2-binary\\npip install workalendar')")


# In[ ]:


import pandas as pd
import geopandas as gpd
import numpy as np
import os

#from sqlalchemy import create_engine
#import env

from pyspark.sql import SparkSession
from pyspark.sql.functions import substring, length, col, expr
from pyspark.sql.types import *

import requests

from datetime import datetime, timedelta, date
import time
import pytz
from workalendar.europe import Netherlands

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

import helpers_gvb as h

import importlib   # to reload helpers without restarting kernel: importlib.reload(h)

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings


# ## Settings

# In[ ]:


# create engine for SQL queries
#engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(env.DATABASE_USERNAME_AZ, 
#                                                            env.DATABASE_PASSWORD_AZ, 
#                                                            "igordb.postgres.database.azure.com", 
#                                                            5432, 
#                                                            "igor"),
#                       connect_args={'sslmode':'require'})


# In[ ]:


# stations to create predictions for
stations = ['Centraal Station', 'Station Zuid', 'Station Bijlmer ArenA']


# In[ ]:


today = pd.to_datetime("today")
today_str = str(today.year) + "-" + str(today.month) + "-" + str(today.day)
covid_url = 'https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/date-range/2020-1-1/' + today_str


# ## Main

# ### 1. Get data

# In[ ]:


print('Start loading raw data') 


# In[ ]:


spark = SparkSession     .builder     .getOrCreate()


# In[ ]:


# Load 2019 GVB data (TEMPORARILY)
gvb_2019_bestemming_raw = spark.read.format("csv").option("header", "true").load("s3a://gvb-gvb/topics/gvb/2020/04/20/Datalab_Reis_Bestemming_Uur_2019.csv", sep = ";").toPandas()
gvb_2019_herkomst_raw = spark.read.format("csv").option("header", "true").load("s3a://gvb-gvb/topics/gvb/2020/04/20/Datalab_Reis_Herkomst_Uur_2019.csv", sep = ";").toPandas()


# In[ ]:


# Load 2020 GVB data (TEMPORARILY)
gvb_2020_bestemming_raw = spark.read.format("csv").option("header", "true").load("s3a://gvb-gvb/topics/gvb/2020/*/*/Datalab_Reis_Bestemming_Uur_2020*.csv", sep = ";").toPandas()
gvb_2020_herkomst_raw = spark.read.format("csv").option("header", "true").load("s3a://gvb-gvb/topics/gvb/2020/*/*/Datalab_Reis_Herkomst_Uur_2020*.csv", sep = ";").toPandas()


# In[ ]:


# Load 2021 GVB data (csv format only) (TEMPORARILY)
gvb_2021_bestemming_raw = spark.read.format("csv").option("header", "true").load("s3a://gvb-gvb/topics/gvb/2021/*/*/Datalab_Reis_Bestemming_Uur_2021*.csv", sep = ";").toPandas()
gvb_2021_herkomst_raw = spark.read.format("csv").option("header", "true").load("s3a://gvb-gvb/topics/gvb/2021/*/*/Datalab_Reis_Herkomst_Uur_2021*.csv", sep = ";").toPandas()


# In[ ]:


# Merge data from above 3 cells (TEMPORARILY)
gvb_bestemming_raw_csv = pd.concat([gvb_2019_bestemming_raw, gvb_2020_bestemming_raw, gvb_2021_bestemming_raw])
gvb_herkomst_raw_csv = pd.concat([gvb_2019_herkomst_raw, gvb_2020_herkomst_raw, gvb_2021_herkomst_raw])


# In[ ]:


# Load GVB data in JSON format
gvb_bestemming_raw_json = spark.read.format("json").option("header", "true").load("s3a://gvb-gvb/topics/gvb/*/*/*/*.json.gz", sep = ",").toPandas()
gvb_herkomst_raw_json = spark.read.format("json").option("header", "true").load("s3a://gvb-gvb/topics/gvb-herkomst/*/*/*/*.json.gz", sep = ",").toPandas()


# In[ ]:


# Load weather data
knmi_obs = spark.read.format("json").load("s3a://knmi-knmi/topics/knmi-observations/*/*/*/*").toPandas()
knmi_pred = spark.read.format("json").option("header", "true").load("s3a://knmi-knmi/topics/knmi/2021/*/*/*.json.gz", sep = ";").toPandas()


# In[ ]:


covid_df_raw = pd.DataFrame(requests.get(url = covid_url).json()['data'])


# In[ ]:


holidays_data_raw = Netherlands().holidays(2019) + Netherlands().holidays(2020) + Netherlands().holidays(2021) 


# In[ ]:


vacations_df = h.get_vacations()


# In[ ]:


events = h.get_events()


# In[ ]:


static_gvb = pd.read_csv('static-gvb.csv')


# ### 2. Prepare data

# #### Pre-process data sources

# In[ ]:


print('Start pre-processing data')


# In[ ]:


bestemming, herkomst = h.merge_csv_json(gvb_bestemming_raw_json, gvb_herkomst_raw_json, gvb_bestemming_raw_json, gvb_herkomst_raw_json)


# In[ ]:


# Cast 'AantalReizen' to int to sum up
bestemming['AantalReizen'] = bestemming['AantalReizen'].astype(int)
herkomst['AantalReizen'] = herkomst['AantalReizen'].astype(int)

# Remove all duplicates
bestemming.drop_duplicates(inplace=True)
herkomst.drop_duplicates(inplace=True)

# Group by station name because we are analysing per station on stop level
bestemming_grouped = bestemming.groupby(['Datum', 'UurgroepOmschrijving (van aankomst)', 'AankomstHalteCode', 'AankomstHalteNaam'], as_index=False)['AantalReizen'].sum()
herkomst_grouped = herkomst.groupby(['Datum', 'UurgroepOmschrijving (van vertrek)', 'VertrekHalteCode', 'VertrekHalteNaam'], as_index=False)['AantalReizen'].sum()


# In[ ]:


bestemming_herkomst = h.merge_bestemming_herkomst_stop_level(bestemming_grouped, herkomst_grouped)


# In[ ]:


bestemming_herkomst_station_types = h.set_station_type(bestemming_herkomst, static_gvb)


# In[ ]:


# Note that the station type is converted to dummies
gvb_dfs = []

for station in stations:
    gvb_dfs.append(pd.get_dummies(h.preprocess_gvb_data_for_modelling(bestemming_herkomst_station_types, station),
                                 columns=['type']))


# In[ ]:


knmi_historical = h.preprocess_knmi_data_hour(knmi_obs)


# In[ ]:


knmi_forecast = h.preprocess_metpre_data(knmi_pred)


# In[ ]:


covid_df = h.preprocess_covid_data(covid_df_raw)


# In[ ]:


holiday_df = h.preprocess_holiday_data(holidays_data_raw)


# #### Merge datasources

# In[ ]:


gvb_dfs_merged = []

for df in gvb_dfs:
    gvb_dfs_merged.append(h.merge_gvb_with_datasources(df, knmi_historical, covid_df, holiday_df, vacations_df, events))


# ### 3. Clean data

# In[ ]:


print('Start cleaning data')


# #### Interpolate missing data

# In[ ]:


def interpolate_missing_values_stop_level(data_to_interpolate):
    df = data_to_interpolate.copy()
    random_state_value = 1 # Ensure reproducability
        
    # Train check-ins interpolator
    checkins_interpolator_cols = ['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-outs', 'type_metro',
                                  'type_tram/bus']
    checkins_interpolator_targets = ['check-ins']
    
    X_train = df.dropna()[checkins_interpolator_cols]
    y_train = df.dropna()[checkins_interpolator_targets]

    checkins_interpolator = RandomForestRegressor(random_state=random_state_value)
    checkins_interpolator.fit(X_train, y_train)
    
    # Train check-outs interpolator
    checkouts_interpolator_cols = ['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-ins', 'type_metro',
                                  'type_tram/bus']
    checkouts_interpolator_targets = ['check-outs']
    
    X_train = df.dropna()[checkouts_interpolator_cols]
    y_train = df.dropna()[checkouts_interpolator_targets]
    
    checkouts_interpolator = RandomForestRegressor(random_state=random_state_value)
    checkouts_interpolator.fit(X_train, y_train)
    
    # Select rows which need interpolation
    df_to_interpolate = df.drop(df.loc[(df['check-ins'].isna()==True) & (df['check-outs'].isna()==True)].index)
    
    # Interpolate check-ins
    checkins_missing = df_to_interpolate[(df_to_interpolate['check-outs'].isna()==False) & (df_to_interpolate['check-ins'].isna()==True)].copy()
    checkins_missing['check-ins'] = checkins_interpolator.predict(checkins_missing[['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-outs', 'type_metro', 'type_tram/bus']])
    
    # Interpolate check-outs
    checkouts_missing = df_to_interpolate[(df_to_interpolate['check-ins'].isna()==False) & (df_to_interpolate['check-outs'].isna()==True)].copy()
    checkouts_missing['check-outs'] = checkouts_interpolator.predict(checkouts_missing[['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-ins', 'type_metro', 'type_tram/bus']])
    
    # Insert interpolated values into main dataframe
    for index, row in checkins_missing.iterrows():
        df.loc[df.index==index, 'check-ins'] = row['check-ins']
        
    for index, row in checkouts_missing.iterrows():
        df.loc[df.index==index, 'check-outs'] = row['check-outs']
        
    return df 


# In[ ]:


gvb_dfs_interpolated = []

for df in gvb_dfs_merged:
    gvb_dfs_interpolated.append(h.interpolate_missing_values_stop_level(df))


# #### Create features

# In[ ]:


gvb_dfs_final = []

for df in gvb_dfs_interpolated:
    df['check-ins'] = df['check-ins'].astype(int)
    df['check-outs'] = df['check-outs'].astype(int)
    df[['check-ins_week_ago', 'check-outs_week_ago']] = df.apply(lambda x: h.get_crowd_last_week_stop_level(df, x), axis=1, result_type="expand")
    gvb_dfs_final.append(df)


# ### 4. Create model dataframes

# In[ ]:


data_splits = []

for df in gvb_dfs_final:
    train, validation, test = h.get_train_val_test_split(df.dropna())
    data_splits.append([train, validation, test])


# In[ ]:


# Define features and targets. This is the same for all stations at the moment.
features = ['year', 'month', 'weekday', 'hour', 'holiday', 'vacation', 'planned_event', 'check-ins_week_ago', 
            'check-outs_week_ago', 'stringency', 'temperature', 'wind_speed', 'precipitation_h', 'type_metro', 'type_tram/bus']
targets = ['check-ins', 'check-outs']


# In[ ]:


X_train_splits = []
y_train_splits = []

X_validation_splits = []
y_validation_splits = []

X_test_splits = []
y_test_splits = []

for split in data_splits:
    X_train_splits.append(split[0][features])
    y_train_splits.append(split[0][targets])
    
    X_validation_splits.append(split[1][features])
    y_validation_splits.append(split[1][targets])
    
    X_test_splits.append(split[2][features])
    y_test_splits.append(split[2][targets])


# In[ ]:


# Dataframes to predict check-ins and check-outs of next week. Note that some changes are made due to the stop level.
X_predict_dfs_metro = []
X_predict_dfs_trambus = []

features_pred = features.copy()
features_pred.remove('type_metro')
features_pred.remove('type_tram/bus')

for df in gvb_dfs_final:
    X_predict_dfs_metro.append(h.get_future_df(features_pred, df, covid_df.tail(1)['stringency'][0], holiday_df, vacations_df, knmi_forecast, events))
    X_predict_dfs_trambus.append(h.get_future_df(features_pred, df, covid_df.tail(1)['stringency'][0], holiday_df, vacations_df, knmi_forecast, events))

for df in X_predict_dfs_metro:
    df['type_metro'] = 1
    df['type_tram/bus'] = 0

for df in X_predict_dfs_trambus:
    df['type_tram/bus'] = 1
    df['type_metro'] = 0


# ### 5. Create model

# In[ ]:


print('Start modelling')


# In[ ]:


basic_models = []

for x in range(0, len(data_splits)):
    model_basic, r_squared_basic, mae_basic, rmse_basic = h.train_random_forest_regressor(X_train_splits[x], y_train_splits[x], 
                                                                                          X_validation_splits[x], y_validation_splits[x], 
                                                                                          None)
    basic_models.append([model_basic, r_squared_basic, mae_basic, rmse_basic])


# #### Tune (hyper-)parameters (not done because models currently do not improve with hyperparameter tuning)

# In[ ]:


# Specify hyperparameters, these could be station-specific. For now, default hyperparameter settings are being used.
centraal_station_hyperparameters = None
station_zuid_hyperparameters = None
station_bijlmer_arena_hyperparameters = None

hyperparameters = [centraal_station_hyperparameters,
                   station_zuid_hyperparameters, 
                   station_bijlmer_arena_hyperparameters]


# In[ ]:


#tuned_models = []

#for x in range(0, len(data_splits)):
#    model_tuned, r_squared_tuned, mae_tuned, rmse_tuned = h.train_random_forest_regressor(X_train_splits[x], y_train_splits[x], 
#                                                                                          X_validation_splits[x], y_validation_splits[x], 
#                                                                                          hyperparameters[x])
#    tuned_models.append([model_tuned, r_squared_tuned, mae_tuned, rmse_tuned])


# ##### Improvements compared to basic model (negative is worse performance)

# In[ ]:


#for x in range(0, len(basic_models)):
#    print("R-squared difference", tuned_models[x][1]-basic_models[x][1])
#    print("MAE difference", tuned_models[x][2]-basic_models[x][2])
#    print("RMSE difference", tuned_models[x][3]-basic_models[x][3])


# #### Train test model (including validation data)

# In[ ]:


test_models = []

for x in range(0, len(data_splits)):
    X_train_with_val = pd.concat([X_train_splits[x], X_validation_splits[x]])
    y_train_with_val = pd.concat([y_train_splits[x], y_validation_splits[x]])
    
    model_test, r_squared_test, mae_test, rmse_test = h.train_random_forest_regressor(X_train_with_val, y_train_with_val, 
                                                                                          X_test_splits[x], y_test_splits[x], 
                                                                                          hyperparameters[x])
    test_models.append([model_test, r_squared_test, mae_test, rmse_test])


# #### Check models on R-squared score

# In[ ]:


for x in range(0, len(test_models)):
    station_name = stations[x]
    r_squared = test_models[x][1]
    if r_squared < 0.7:
        warnings.warn("Model for " + station_name + " shows unexpected performance!")


# #### Train final models (to make predictions)

# In[ ]:


final_models = []

for x in range(0, len(data_splits)):
    X_train_with_val = pd.concat([X_train_splits[x], X_validation_splits[x], X_test_splits[x]])
    y_train_with_val = pd.concat([y_train_splits[x], y_validation_splits[x], y_test_splits[x]])
    
    model_final = h.train_random_forest_regressor(X_train_with_val, y_train_with_val, X_test_splits[x], y_test_splits[x], 
                                                  hyperparameters[x])[0]
    final_models.append(model_final)


# ### 6. Prepare output

# In[ ]:


print('Start preparing data')


# In[ ]:


predictions_metro = []

for predict_df in X_predict_dfs_metro:
    for model in final_models:
        prediction = h.predict(model, predict_df)
        predictions_metro.append(prediction)


# In[ ]:


predictions_trambus = []

for predict_df in X_predict_dfs_trambus:
    for model in final_models:
        prediction = h.predict(model, predict_df)
        predictions_trambus.append(prediction)


# In[ ]:


# Calculate crowd levels here? Or use trigger in database?


# ### 7. Store data

# In[ ]:


print('Start storing data')


# In[ ]:


for x in range(0, len(stations)):
    station_name = stations[x] # Use this to write predictions to database
    
    metro_predictions_for_station = predictions_metro[x].drop(columns=['year', 'month', 'weekday', 'holiday', 'vacation', 
                                                                       'planned_event', 'check-ins_week_ago', 
                                                                       'check-outs_week_ago', 'stringency', 'temperature', 
                                                                       'wind_speed', 'precipitation_h', 'type_tram/bus',
                                                                       'type_metro'])
    metro_predictions_for_station['type'] = 'metro'
    
    
    trambus_predictions_for_station = predictions_trambus[x].drop(columns=['year', 'month', 'weekday', 'holiday', 'vacation', 
                                                                       'planned_event', 'check-ins_week_ago', 
                                                                       'check-outs_week_ago', 'stringency', 'temperature', 
                                                                       'wind_speed', 'precipitation_h', 'type_tram/bus',
                                                                       'type_metro'])
    trambus_predictions_for_station['type'] = 'tram/bus'
    
    # Add time to datetime column so that it is easier to store in database
    final_prediction_dataframe = pd.concat([metro_predictions_for_station, trambus_predictions_for_station])
    final_prediction_dataframe['datetime'] = final_prediction_dataframe.apply(lambda x: x['datetime'].replace(hour=x['hour']), axis=1)
    final_prediction_dataframe.drop(columns=['hour'], inplace=True)
    final_prediction_dataframe['Station'] = station_name
    
    ### Code to write data to database


# In[ ]:


print('Finished storing data')

