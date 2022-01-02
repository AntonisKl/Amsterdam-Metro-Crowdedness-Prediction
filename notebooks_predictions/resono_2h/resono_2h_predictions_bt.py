# Resono 2h predictions

# ### Preparations

# Main libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import copy
import prediction_model_helpers_bt as h

# Database connection
from sqlalchemy import create_engine, inspect

from pyspark.sql import SparkSession
from pyspark.sql.functions import substring, length, col, expr
from pyspark.sql.types import *
import requests

# Dates
from datetime import datetime, timedelta, date
import pytz
from workalendar.europe import Netherlands

# Plotting
import matplotlib
import matplotlib.pyplot as plt

def prepare_data(env, freq, predict_period, n_samples_day, Y_names, target, start_learnset):
    '''
    Prepare data for Resono predictions 
    '''
    
    # ### 1. Get data
    
    # Database connection
    engine_azure, table_names = h.prepare_engine(env)
    
    # Which locations to include (based on druktebeeld inclusion)
    if Y_names == "all":
        Y_inc = select_locations(engine_azure)

    # Read in raw Resono data
    resono_df_raw = h.get_data(engine_azure, "ingested.resono", Y_names, start_learnset)
    
    # Store thresholds
    thresholds = get_thresholds(engine_azure, Y_names) 
    
    # Select moment to end operational forecast based on last value of variable to predict in database
    start_prediction = pd.date_range(resono_df_raw["measured_at"].max(), periods = 2, freq = freq)[1]
    end_prediction = pd.date_range(start_prediction, periods = predict_period, freq = freq)
    end_prediction = end_prediction[len(end_prediction)-1]

    # COVID data raw
    covid_url = 'https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/date-range/'+ str(start_learnset) + "/" + str(start_prediction)
    covid_df_raw = pd.DataFrame(requests.get(url = covid_url).json()['data'])

    # Holidays data raw
    holidays_data_raw = Netherlands().holidays(2020) + Netherlands().holidays(2021) 
    
    # ### 2. Preprocess data

    # Resono data preprocessed
    resono_df = h.preprocess_resono_data(resono_df_raw, freq, end_prediction)

    if Y_names == "all":
    # Select locations that are included in druktebeeld
        resono_df = resono_df[resono_df.columns[resono_df.columns.isin(Y_inc)]]
    
    # COVID data preprocessed
    covid_df = h.preprocess_covid_data(covid_df_raw, freq, end_prediction)

    # Holiday data preprocessed
    holiday_df = h.preprocess_holidays_data(holidays_data_raw, freq, end_prediction)
    
    # Join location-independent data into base df
    base_df = covid_df.join(holiday_df)
    
    # List of all  (included) location names
    Y_names_all = list(resono_df.columns)

    return base_df, resono_df, resono_df_raw, start_prediction, end_prediction, thresholds, Y_names_all


def get_resono_predictions(df, resono_df_raw, freq, predict_period, n_samples_day, n_samples_week, Y_name, data_source, 
                           target, outlier_removal, start_learnset, use_smote,
                           current_model_version, current_data_version, start_prediction, end_prediction, thresholds):
    '''
    Get 2h predictions for one Resono location.
    '''
    
    # ### 1. Clean data

    # Impute/drop missing data and substitute outliers
    df = h.clean_data(df, target, Y_name, n_samples_day, cols_to_clean = None, outlier_removal = outlier_removal)
    
    if target == "level":
        # Re-calculate crowd levels
        df = h.get_crowd_levels(df, Y_name, thresholds) 

    # Add time features
    df = h.add_time_variables(df)

    # Create new features from the data
    df = h.add_lag_variables(df, Y_name, target, predict_period, n_samples_day, n_samples_week)

    # filter data based on start learnset
    df = df[start_learnset:]

    # drop time slots for which missing values remain
    df = df.dropna()
    
    # No predictions if there is no training data 
    if df.empty:
        print("No predictions made for " + Y_name + ": not enough training data.")
        
        df_y_predict = h.get_future_df(start_prediction, predict_period, freq)
        
        y_scaler = None
        thresholds_scaled = None

    else:
        # scale dataset
        df_unscaled = df.copy()
        df, y_scaler = h.scale_variables(df, Y_name, target, method = "standard")
        
        # scale thresholds
        thresholds_scaled = copy.deepcopy(thresholds)
        thresholds_scaled = thresholds_scaled[Y_name]
        
        # if there are no samples with crowd level 1.0, the scaled thresholds will be identical
        thresholds_scaled[1] = df[Y_name][(np.abs(df_unscaled[Y_name] - thresholds[Y_name][1])).argmin()]
        thresholds_scaled[2] = df[Y_name][(np.abs(df_unscaled[Y_name] - thresholds[Y_name][2])).argmin()]

    # ### 2. Create model dataframes

        df_X_train, df_y_train = h.get_train_df(df, Y_name, start_prediction) 

        df_y_predict = h.get_future_df(start_prediction, predict_period, freq)

        df_X_predict = df.drop(Y_name, 1)

        # Select features for prediction period
        df_X_predict = df_X_predict[start_prediction:end_prediction]


    # ### 3. Create operational prediction

        # Linear regression model with L2-regularization (ridge)
        model = h.train_model_ridge_regression(df_X_train, df_y_train, Y_name, target, 
                                               thresholds_one = thresholds_scaled, use_smote = use_smote)

        df_y_predict[Y_name] = h.test_model_ridge_regression(model, df_X_predict)
                                                         
        # unscale prediction
        if target == "count":
            df_y_predict = h.unscale_y(df_y_predict, y_scaler)


    # ### 4. Prepare output

    final_df = h.prepare_final_dataframe(df_y_predict, resono_df_raw, data_source, target, current_model_version,
                                         current_data_version)

    return df, final_df, y_scaler, thresholds_scaled


def select_locations(engine):
    '''
    Select the locations to make predictions for based on inclusion in druktebeeld.
    '''
    
    # Select all locations
    query = "SELECT * FROM ingested.static_resono_locations LIMIT 3000000"
    resono_static_df = pd.read_sql_query(query, con = engine)
    
    # Select locations to include
    Y_inc = list(resono_static_df[resono_static_df['include_druktebeeld'] == 1]['location'])

    return Y_inc


def get_thresholds(engine, names):
    '''
    Get the thresholds for one or more Resono locations.
    '''
    
    # Select all locations
    if names == 'all':
        query = "SELECT * FROM ingested.static_resono_locations LIMIT 3000000"
        resono_static_df = pd.read_sql_query(query, con = engine)
        
        # Save thresholds per location
        thresholds = dict()
        for name in resono_static_df['location'].unique():
            thresholds[name] = list_thresholds(name, resono_static_df)
        
    # Select one location
    elif isinstance(names, str):
        query = "SELECT * FROM ingested.static_resono_locations WHERE location = '" + names + "' LIMIT 3000000".format(names)
        resono_static_df = pd.read_sql_query(query, con = engine)
        
        # Save thresholds per location
        thresholds = dict()
        thresholds[names] = list_thresholds(names, resono_static_df)

        
    # Select locations out of list of location names
    else:
        names = tuple(names)
        query = "SELECT * FROM ingested.static_resono_locations WHERE location IN {} LIMIT 3000000".format(names)
        resono_static_df = pd.read_sql_query(query, con = engine)
    
        
        # Save thresholds per location
        thresholds = dict()
        for name in names:
            thresholds[name] = list_thresholds(name, resono_static_df)
    
    return thresholds


def list_thresholds(name, resono_static_df):
    ''' 
    List the thresholds for one location.
    '''
    
    th = [float('-inf'), resono_static_df[resono_static_df['location'] == name]['crowd_threshold_low'].values[0], 
          resono_static_df[resono_static_df['location'] == name]['crowd_threshold_high'].values[0], float('inf')]
    
    return th


def get_location_df(base_df, resono_df, Y_name):
    '''
    Get dataframe with target and predictor variables for one location.
    '''
    
    df = pd.concat([resono_df[Y_name], base_df], 1)
    
    return df
    


