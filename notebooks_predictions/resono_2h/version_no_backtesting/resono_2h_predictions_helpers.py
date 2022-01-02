# Resono 2h predictions helper functions

# ### Preparations

# General
import pandas as pd
import numpy as np
import os
import copy

# Database connection
from sqlalchemy import create_engine, inspect
from pyspark.sql import SparkSession
from pyspark.sql.functions import substring, length, col, expr
from pyspark.sql.types import *
import requests

# Dates
from datetime import datetime, timedelta, date
import datetime as dt
import pytz
from workalendar.europe import Netherlands

# Modelling
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
import math

# Cleaning
from scipy.optimize import curve_fit
from sklearn import preprocessing, svm

# Sample weights
from sklearn.utils.class_weight import compute_sample_weight

# Synthetic minority oversampling
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE

### FUNCTION - prepare data

def prepare_data(env, freq, predict_period, n_samples_day, Y_names, target, start_learnset):
    '''
    Prepare data for Resono predictions 
    '''
    
    # ### 1. Get data
    
    # Database connection
    engine_azure, table_names = prepare_engine(env)
    
    # Which locations to include (based on druktebeeld inclusion)
    if Y_names == "all":
        Y_inc = select_locations(engine_azure)

    # Read in raw Resono data
    resono_df_raw = get_data(engine_azure, "ingested.resono", Y_names, start_learnset)
    
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
    resono_df = preprocess_resono_data(resono_df_raw, freq, end_prediction)

    if Y_names == "all":
    # Select locations that are included in druktebeeld
        resono_df = resono_df[resono_df.columns[resono_df.columns.isin(Y_inc)]]
    
    # COVID data preprocessed
    covid_df = preprocess_covid_data(covid_df_raw, freq, end_prediction)

    # Holiday data preprocessed
    holiday_df = preprocess_holidays_data(holidays_data_raw, freq, end_prediction)
    
    # Join location-independent data into base df
    base_df = covid_df.join(holiday_df)
    
    # List of all  (included) location names
    Y_names_all = list(resono_df.columns)

    return base_df, resono_df, resono_df_raw, start_prediction, end_prediction, thresholds, Y_names_all


### FUNCTION - get predictions

def get_resono_predictions(df, resono_df_raw, freq, predict_period, n_samples_day, n_samples_week, Y_name, data_source, 
                           target, outlier_removal, start_learnset, use_smote,
                           current_model_version, current_data_version, start_prediction, end_prediction, thresholds):
    '''
    Get 2h predictions for one Resono location.
    '''
    
    # ### 1. Clean data

    # Impute/drop missing data and substitute outliers
    df = clean_data(df, target, Y_name, n_samples_day, cols_to_clean = None, outlier_removal = outlier_removal)
    
    if target == "level":
        # Re-calculate crowd levels
        df = get_crowd_levels(df, Y_name, thresholds) 

    # Add time features
    df = add_time_variables(df)

    # Create new features from the data
    df = add_lag_variables(df, Y_name, target, predict_period, n_samples_day, n_samples_week)

    # filter data based on start learnset
    df = df[start_learnset:]

    # drop time slots for which missing values remain
    df = df.dropna()
    
    # No predictions if there is no training data 
    if df.empty:
        print("No predictions made for " + Y_name + ": not enough training data.")
        
        df_y_predict = get_future_df(start_prediction, predict_period, freq)
        
        y_scaler = None
        thresholds_scaled = None

    else:
        # scale dataset
        df_unscaled = df.copy()
        df, y_scaler = scale_variables(df, Y_name, target, method = "standard")
        
        # scale thresholds
        thresholds_scaled = copy.deepcopy(thresholds)
        thresholds_scaled = thresholds_scaled[Y_name]
        
        # if there are no samples with crowd level 1.0, the scaled thresholds will be identical
        thresholds_scaled[1] = df[Y_name][(np.abs(df_unscaled[Y_name] - thresholds[Y_name][1])).argmin()]
        thresholds_scaled[2] = df[Y_name][(np.abs(df_unscaled[Y_name] - thresholds[Y_name][2])).argmin()]

    # ### 2. Create model dataframes

        df_X_train, df_y_train = get_train_df(df, Y_name, start_prediction) 

        df_y_predict = get_future_df(start_prediction, predict_period, freq)

        df_X_predict = df.drop(Y_name, 1)

        # Select features for prediction period
        df_X_predict = df_X_predict[start_prediction:end_prediction]


    # ### 3. Create operational prediction

        # Linear regression model with L2-regularization (ridge)
        model = train_model_ridge_regression(df_X_train, df_y_train, Y_name, target, 
                                               thresholds_one = thresholds_scaled, use_smote = use_smote)

        df_y_predict[Y_name] = test_model_ridge_regression(model, df_X_predict)
                                                         
        # unscale prediction
        if target == "count":
            df_y_predict = unscale_y(df_y_predict, y_scaler)


    # ### 4. Prepare output

    final_df = prepare_final_dataframe(df_y_predict, resono_df_raw, data_source, target, current_model_version,
                                         current_data_version)

    return df, final_df, y_scaler, thresholds_scaled


### FUNCTIONS - pre modelling

def get_start_learnset(train_length = None, date_str = None):
    """
    Get a datetime string for the starting date for training
    
    One of:
    train_length: integer indicating the number of weeks used for training
    date_str: str indicating a starting date (in the format "2020-01-01 00:00:00")
    """
    
    if train_length:
        date_time_current = datetime.now()
        start_learnset = date_time_current - dt.timedelta(days = train_length*7)

    elif date_str:
        start_learnset = pd.to_datetime(date_str)

    return start_learnset


def prepare_engine(cred):
    '''
    Prepare engine to read in data.
    
    cred: the env object with credentials.
    
    Returns the engine and the table names.
    '''
    
    # Create engine object
    engine_azure = create_engine("postgresql://{}:{}@{}:{}/{}".format(cred.DATABASE_USERNAME,
                                                            cred.DATABASE_PASSWORD,
                                                            "igordb.postgres.database.azure.com",
                                                            5432,
                                                            "igor"),
                       connect_args={'sslmode':'require'})
    
    # Check available tables
    inspector = inspect(engine_azure)

    table_names = []
    for table_name in inspector.get_table_names('ingested'):
        table_names.append(table_name)
    
    return engine_azure, table_names


def get_data(engine, table_name, names, start_learnset):
    """
    Read in Resono, CMSA or parking data from the database.
    
    table_name: name of the database table
    names: list of one or more location names (have to match location_name column in table); or 'all' for all locations
    start_learnset: date indicating the moment to begin using data for training. 
    """
    
    # Read in data from two weeks before starting date learn set (necessary for lag features)
    start_date = start_learnset - dt.timedelta(days = 14)
    start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
    
    # Select all locations
    if names == 'all':
        query = "SELECT * FROM " + table_name + " WHERE measured_at >= '" + start_date + "' LIMIT 3000000"
        df_raw = pd.read_sql_query(query, con = engine)
        
    # Select one location
    elif isinstance(names, str):
        query = "SELECT * FROM " + table_name + " WHERE location_name = '" + names + "' AND measured_at >= '".format(names) + start_date + "' LIMIT 3000000"
        df_raw = pd.read_sql_query(query, con = engine)
        
    # Select locations out of list of location names
    else:
        names = tuple(names)
        query = "SELECT * FROM " + table_name + " WHERE location_name IN {} AND measured_at >= '".format(names) + start_date + "' LIMIT 3000000"
        df_raw = pd.read_sql_query(query, con = engine)
    
    return df_raw


def preprocess_resono_data(df_raw, freq, end_prediction):
    """
    Prepare the raw resono data for modelling. 
    """
    
    # Drop duplicates
    df = df_raw.copy()
    df = df.drop_duplicates() 
    
    # Fix timestamp
    df["datetime"] = pd.to_datetime(df["measured_at"])
    df = df.sort_values(by = "datetime", ascending = True)
        
    # Wide format
    df = df.pivot_table(index = ["datetime"], columns = "location_name", values = "total_count").reset_index()
    df = df.set_index('datetime')
        
    # Change column names
    df.rename(columns = {'location_name': 'location'})
    
    # Set right sampling frequency
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  # Any new samples are treated as missing data
    
    return df


def preprocess_covid_data(df_raw, freq, end_prediction):
    """
    Prepare the raw covid stringency data for modelling. 
    """
    
    # Put data to dataframe
    df_raw_unpack = df_raw.T['NLD'].dropna()
    df = pd.DataFrame.from_records(df_raw_unpack) 
    
    # Add datetime column
    df['datetime'] = pd.to_datetime(df['date_value'])
    
    # Select columns
    df_sel = df[['datetime', 'stringency']]
       
    # extend dataframe to 14 days in future (based on latest value)
    dates_future = pd.date_range(df['datetime'].iloc[-1], periods = 14, freq='1d')
    df_future = pd.DataFrame(data = {'datetime': dates_future, 
                                       'stringency': df['stringency'].iloc[-1]})
    
    # Add together and set index
    df_final = df_sel.append(df_future.iloc[1:])
    df_final = df_final.set_index('datetime')
    
    # Set right sampling frequency
    idx = pd.date_range(df_final.index[0], end_prediction, freq = freq)
    df_final = df_final.reindex(idx)  
    
    # Fill missing values with nearest value
    df_final = df_final.fillna(method = "ffill")
    df_final = df_final.fillna(method = "bfill")
 
    return df_final


def preprocess_holidays_data(holidays, freq, end_prediction):
    """
    Prepare the raw holiday data for modelling. 
    """
    
    # Put in dataframe
    holiday_df = pd.DataFrame(holidays).rename(columns = {0: 'date', 1: 'holiday'})
    
    # Create datetime index
    holiday_df['datetime'] = pd.to_datetime(holiday_df['date'])
    holiday_df = holiday_df.set_index('datetime')
    
    # Create dummy variable
    holiday_df['holiday_dummy'] =  1
    holiday_df_d = holiday_df.resample('1d').asfreq() # dataframe with all days
    holiday_df_d['holiday_dummy'] = holiday_df_d['holiday_dummy'].fillna(0)
    holiday_df_d['holiday_dummy'] = holiday_df_d['holiday_dummy'].astype(int)
    
    # Select column
    holiday_df_d = holiday_df_d[['holiday_dummy']]
    
    # Set right sampling frequency
    idx = pd.date_range(holiday_df_d.index[0], end_prediction, freq = freq)
    holiday_df_d = holiday_df_d.reindex(idx)  
    
    # Fill missing values with nearest value
    holiday_df_d = holiday_df_d.fillna(method = "ffill")
    
    # set back to right dtype
    holiday_df_d['holiday_dummy'] = holiday_df_d['holiday_dummy'].astype(int)
    
    return holiday_df_d


def get_crowd_levels(df, Y_name, thresholds_all = None, thresholds_one = None):
    '''
    Recalculate the crowd levels based on the thresholds from the static Resono/CMSA table.
    '''
    
    df_level = df.copy()
    
    # Re-calculate crowd levels
    if thresholds_all is not None:
        th = thresholds_all[Y_name]
    elif thresholds_one is not None:
        th = thresholds_one
        
    # If the scaled thresholds are identical (no samples with crowd level 1.0), only use the first threshold
    if (th[1] == th[2]) | (len(th) == 3):
        if th[1] == th[2]:
            th.pop(2)
        labels = [-1.0, 0.0]
        df_level[Y_name] = pd.cut(df_level[Y_name], bins = th, labels = labels)
    else:
        labels = [-1.0, 0.0, 1.0]
        df_level[Y_name] = pd.cut(df_level[Y_name], bins = th, labels = labels)
    
    df_level[Y_name] = df_level[Y_name].astype('category')
    
    return df_level


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


### FUNCTIONS - cleaning
    
def clean_data(df, target, Y_name, n_samples_day, cols_to_clean, outlier_removal, nu = 0.2, gamma = 0.1):
    """
    Clean data by imputing missing values and removing outliers (replacing with interpolation/extrapolation).
    Days that are fully missing are dropped from the dataframe (to prevent strange interpolation results).
    
    cols_to_clean: column names of columns to clean
    outlier_removal: "yes" or "no"
    # nu: value for nu parameter for outlier removal model (default is 0.2) 
    # gamma: value for gamma parameter for outlier removal model (default = 0.1)
    
    Returns the dataframe with missing values interpolated and outliers replaced.
    """
    
    # Initialize data frame to clean
    df_to_clean = df.copy()
    # If target variable is count, add to columns to clean
    if cols_to_clean is not None:
        cols = cols_to_clean.copy()
        if target == 'count':
            cols.append(Y_name)
    else:
        cols = Y_name
        
    # Define date column to group on 
    df_to_clean['date'] = df_to_clean.index.date
    
    # Find missing days for target column
    dates_to_drop = []
    # Index of fully missing days
    day_missing_idx = np.where(df_to_clean[Y_name].isnull().groupby(df_to_clean['date']).sum() == n_samples_day)[0]
    # Find the dates
    dates_to_drop.extend(df_to_clean['date'].unique()[day_missing_idx])

    # Select columns to impute
    df_to_impute = df_to_clean[[cols]]
    
    # First interpolate/extrapolate missing values 
    df_to_impute = interpolate_ts(df_to_impute)
    df_to_impute = extrapolate_ts(df_to_impute)
    
    if outlier_removal == "yes":
        # Outlier detection and replacement
        SVM_models = create_SVM_models(df_to_impute)
        df_to_impute = SVM_outlier_detection(df_to_impute, SVM_models)
        df_to_impute = interpolate_ts(df_to_impute)
        df_to_impute = extrapolate_ts(df_to_impute)
    
    # Put cleaned variables in data frame
    df_cleaned = df_to_clean.copy()
    df_cleaned[cols] = df_to_impute.copy()
    df_cleaned.index.name = 'datetime'
    
    # Drop the dates that are fully mising
    df_cleaned = df_cleaned[~(df_cleaned['date'].isin(dates_to_drop))]
     
    # Drop date column
    df_cleaned = df_cleaned.drop('date', 1)
    
    # Use forward fill imputation if the target is categorical
    if target == "level":
        df_cleaned[Y_name] = df_cleaned[Y_name].fillna(method = "ffill")

    return df_cleaned
    

def interpolate_ts(df, min_samples = 2):
    """
    Interpolate missing values. 
    
    df: dataframe to interpolate
    min_samples: minimum number of samples necessary to perform interpolation (default = 2)
    
    Returns the dataframe with missing values interpolated.
    """
    
    # Initialize new dataframe
    df_ip = df.copy()
        
    # For each location
    for idx, location in enumerate(df.columns):
        
        # Initialize new location data
        ts = df.iloc[:, idx]
        ts_ip = ts.copy()
        
        # Only interpolate if there are enough data points
        if ts_ip.count() >= min_samples:
            
            # Interpolate missing values
            ts_ip = ts_ip.interpolate(method = 'cubicspline', limit_area = 'inside')
        
        df_ip.iloc[:, idx] = ts_ip
        
        # No negative values for counts
        df_ip = df_ip.clip(lower = 0)
        
    return df_ip


def extrapolate_ts(df, min_samples = 1):
    """
    Extrapolate missing values.
    
    df: dataframe to extrapolate
    min_samples: minimum number of samples necessary to perform extrapolation (default = 1)
    
    Returns the dataframe with missing values extrapolated.
    """
    
    # Initialize new dataframe
    df_ep = df.copy()
        
    # For each location
    for idx, location in enumerate(df.columns):
        
        # Initialize new location data
        ts = df.iloc[:, idx]
        ts_ep = ts.copy()
        
        # Only extrapolate if there are enough data points
        if ts_ep.count() >= min_samples:

            # Temporarily remove dates and make index numeric
            index = ts.index
            ts_temp = ts_ep.reset_index()
            ts_temp = ts_temp.iloc[:, 1]
    
            # Function to curve fit to the data (3rd polynomial)
            def func(x, a, b, c, d):
                return a * (x ** 3) + b * (x ** 2) + c * x + d

            # Initial parameter guess, just to kick off the optimization
            guess = (0.5, 0.5, 0.5, 0.5)

            # Create copy of data to remove NaNs for curve fitting
            ts_fit = ts_temp.dropna()

            # Curve fit 
            x = ts_fit.index.astype(float).values
            y = ts_fit.values
    
            # Curve fit column and get curve parameters
            params = curve_fit(func, x, y, guess)

            # Get the index values for NaNs in the column
            x = ts_temp[pd.isnull(ts_temp)].index.astype(float).values
        
            # Extrapolate those points with the fitted function
            ts_temp[x] = func(x, * params[0])
    
            # Put date index back
            ts_temp.index = index
            ts_ep = ts_temp.copy()
        
            df_ep.iloc[:, idx] = ts_ep
            
    # No negative values for counts
    df_ep = df_ep.clip(lower = 0)
        
    return df_ep


def create_SVM_models(df, min_samples = 1, nu = 0.05, gamma = 0.01):
    """
    Create one-class SVM model for each variable to perform outlier removal.
    
    # df: data frame to perform outlier removal on
    # min_samples: minimum number of samples needed (for each variable) to create a SVM model (default = 1)
    # nu: value for nu parameter (default is 0.05) 
    # gamma: value for gamma parameter (default = 0.01)
    
    # Returns a list of SVM models for each variable.
    """
    
    # Initialize list of models for each location
    SVM_models = []

    # For each location
    for idx, location in enumerate(df.columns):
        
        # Select location and fit one-class SVM 
        ts = df.iloc[:, idx]
        
        # Only create a model if there are enough data points
        if ts.count() >= min_samples:
        
            scaler = preprocessing.StandardScaler()
            ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
            model = svm.OneClassSVM(nu = nu, kernel = "rbf", gamma = gamma)
            model.fit(ts_scaled) 
        
            # Save the model
            SVM_models.append(model)
        
        # Otherwise add None to the list of models
        else:
            SVM_models.append(None)
        
    return SVM_models

def SVM_outlier_detection(df, SVM_models):  
    """
    Detects outliers for each variable in the dataframe.
    
    df: dataframe to apply outlier detection on using SVM models
    SVM_models: list of SVM models (one for each variable)
    
    Returns a dataframe with the outliers replaced by NaNs.
    """
    
    # Initialize new dataframe
    df_detected = df.copy()
        
    # For each location
    for idx, location in enumerate(df.columns):
        
        # Initialize new location data
        ts = df.iloc[:, idx]
        ts_detected = ts.copy()
    
        # Detect outliers using the one-class SVM model for this location
        if SVM_models[idx] is not None and ts.isnull().sum().sum() == 0:
            scaler = preprocessing.StandardScaler()
            ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
            pred = SVM_models[idx].predict(ts_scaled)
            to_idx = lambda x: True if x==-1 else False
            v_to_idx = np.vectorize(to_idx)
            outliers_idx = v_to_idx(pred)
        
            # Set outliers to NaN
            # If not all data points have been marked as outliers; otherwise do not clean
            if outliers_idx.sum() != len(ts_detected):
                ts_detected[outliers_idx] = np.nan
    
        df_detected.iloc[:, idx] = ts_detected
        
        df_detected.index.name = 'datetime'
        
    return df_detected

def get_train_df(df, Y_name, start_prediction):
    """
    Split dataframe into X and y sets for training data
    """
    
    df_X_train = df.drop(Y_name, 1)[:start_prediction].iloc[:-1]
    df_y_train = df[[Y_name]][:start_prediction].iloc[:-1]
    
    return df_X_train, df_y_train


def get_future_df(start_pred, predict_period, freq):
    """
    Create empty data frame for predictions of the target variable for the specfied prediction period
    """
    
    datetime_predict = pd.date_range(start_pred, periods = predict_period, freq = freq)
    df = pd.DataFrame(data = {'datetime' : datetime_predict}).set_index('datetime')
    
    return df


def add_time_variables(df):
    """
    Create dummy variables from weekday, weekend and hour and add these to the dataframe. 
    Also, add cos and sine times
    """
    
    df = df.reset_index()
    
    # add weekday and hour dummies
    df['weekday'] = pd.Categorical(df['datetime'].dt.weekday)
    df['hour'] =  pd.Categorical(df['datetime'].dt.hour)
    
    weekday_dummies = pd.get_dummies(df[['weekday']], prefix='weekday_')
    hour_dummies = pd.get_dummies(df[['hour']], prefix='hour_')
    
    df_time = df.merge(weekday_dummies, left_index = True, right_index = True)
    df_time = df_time.merge(hour_dummies, left_index = True, right_index = True)
    
    # add cyclical time features
    df_time['minutes'] = df_time['datetime'].dt.hour * 60 + df_time['datetime'].dt.minute
    df_time['sin_time'] = np.sin(2 * np.pi * df_time['minutes'] / (24 * 60)) 
    df_time['cos_time'] = np.cos(2 * np.pi * df_time['minutes'] / (24 * 60)) 
        
    df_time = df_time.set_index('datetime')
    
    return df_time


def add_lag_variables(df, Y_name, target, predict_period, n_samples_day, n_samples_week):
    """
    Add lag variables (features that are lagged version of the target variable).
    """

    df[Y_name + "_prev_2h"] = df[Y_name].shift(predict_period)
    df[Y_name + "_prev_day"] = df[Y_name].shift(n_samples_day)
    df[Y_name + "_prev_week"] = df[Y_name].shift(n_samples_week)

    if target == 'count':
        df[Y_name + "_prev_2h_mean_diff"] = df[Y_name + "_prev_2h"] - df[Y_name].mean()
        df[Y_name + "_prev_2h_diff_size"] = df[Y_name] - df[Y_name].shift(predict_period+1)
            
    return df


def scale_variables(df_unscaled, Y_name, target, method):
    """
    Scale the variables and store the scaler object for the target variable.
    
    method: "standard" or "minmax" 
    """

    if target == 'count':
        # Select target variable
        Y_unscaled = df_unscaled[Y_name].values
        Y_unscaled = Y_unscaled.reshape(-1, 1)
    
    # Select continuous columns (do not scale binary/category variables)
    cont_idx = df_unscaled.select_dtypes('float').columns.tolist()
    df_unscaled_cont = df_unscaled.loc[:, cont_idx]
    
    # Standardization
    if method == "standard":
        scaler = preprocessing.StandardScaler().fit(df_unscaled_cont.values)
        
        if target == "count":
            # Store scaler object for target variable
            Y_scaler = preprocessing.StandardScaler().fit(Y_unscaled)
        
    # Min-max scaling
    elif method == "minmax":
        scaler = preprocessing.MinMaxScaler().fit(df_unscaled_cont.values)
        
        if target == "count":
            # Store scaler object for target variable
            Y_scaler = preprocessing.MinMaxScaler().fit(Y_unscaled)
        
    # Scale variables
    df_scaled_cont = scaler.transform(df_unscaled_cont.values)
    df_scaled = df_unscaled.copy()
    df_scaled.loc[:, cont_idx] = df_scaled_cont

    # Convert back to right format
    df_scaled = pd.DataFrame(df_scaled, columns = df_unscaled.columns, index = df_unscaled.index)
    df_scaled.index.name = 'datetime'
    
    if target == "level":
        Y_scaler = None
    
    return df_scaled, Y_scaler


### FUNCTIONS - modelling

def train_model_ridge_regression(df_X_train, df_y_train, Y_name, target, thresholds_all = None, 
                                 thresholds_one = None, use_smote = False):     
    
    """
    Train linear regression model using L2-regularization.
    
    thresholds_scaled: scaled version of the thresholds that matches the scaled target variable 
    use_smote: True/False (synthetic minority oversampling)
    """
    
    if use_smote:
        # Perform synthetic minority oversampling
        if thresholds_all is not None:
            X_train, y_train = perform_smote(df_X_train, df_y_train, Y_name, target, thresholds_all = thresholds_all)
        elif thresholds_one is not None:
            X_train, y_train = perform_smote(df_X_train, df_y_train, Y_name, target, thresholds_one = thresholds_one)
    else:
        # Convert data to numpy array
        X_train = np.array(df_X_train)
        y_train = np.array(df_y_train)
        
    # Initialize model
    model = Ridge()
      
    # Fit model
    model.fit(X_train, y_train)
    
    return model


def test_model_ridge_regression(model, df_X_test):
    """
    Run trained linear regression model using L2-regularization on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(df_X_test)
    
    return prediction


def perform_smote(df_X_train, df_y_train, Y_name, target, thresholds_all = None, thresholds_one = None):
    
    # Convert predictor variables to numpy array
    X_train = np.array(df_X_train)

    # Column indices of categorical features
    cat_index = np.array(df_X_train.dtypes != 'float64')
    
    if target == 'count':
        if thresholds_all is not None:
            y_train = get_crowd_levels(df_y_train, Y_name, thresholds_all = thresholds_all)
        if thresholds_one is not None:
            y_train = get_crowd_levels(df_y_train, Y_name, thresholds_one = thresholds_one)
        y_train_both = pd.concat([df_y_train[Y_name], y_train], 1)
        y_train_both.columns = ['count', 'level']
        avg_count = y_train_both.groupby('level')['count'].mean()
    elif target == 'level':
        # Convert to category dtype (if not done before)
        y_train = df_y_train[Y_name].astype('category')

    # Only oversample if there are at least 3 samples
    if all(y_train.value_counts() >= 3):
        
        # If no categorical predictors: regular SMOTE
        if sum(cat_index) == 0:
            sm = SMOTE(k_neighbors=2) 
            X_train, y_train = sm.fit_resample(X_train, y_train) 
                 
        # If there are also categorical predictors: SMOTENC
        elif sum(cat_index) != X_train.shape[1]:
            sm = SMOTENC(categorical_features = cat_index, k_neighbors=2) 
            X_train, y_train = sm.fit_resample(X_train, y_train) 
        
        if target == 'count':
            # Convert target variable back to continuous value (mean of level)
            y_train = np.where(y_train[len(df_y_train):] == -1.0, 
                                avg_count[0], np.where(y_train[len(df_y_train):] == 0.0,
                                avg_count[1], avg_count[2]))
            y_train = pd.Series(np.concatenate([np.array(df_y_train[Y_name]), y_train.reshape(len(y_train))]).astype('float'))
            
        # Randomly select part of the new samples to dampen the oversampling
        drop_idx = y_train[len(df_y_train):].sample(frac = (1-0.5)).index.values
        y_train = np.delete(np.array(y_train), drop_idx)
        X_train = np.delete(X_train, drop_idx, axis = 0)
                      
    else:
        print("Warning: SMOTE not used because there are not enough samples for all crowd levels (at least 3 per level).")
        
        # Convert target variable to numpy array
        y_train = np.array(df_y_train)
        
    return X_train, y_train

    
### FUNCTIONS - post operational modelling

def unscale_y(y_scaled, Y_scaler):
    """
    Undo scaling on the target variable.
    """
    
    # Undo scaling using the scaler object
    y_unscaled = Y_scaler.inverse_transform(y_scaled)
    
    # Convert back to right format
    df_y_unscaled = pd.DataFrame(y_unscaled, columns = y_scaled.columns, index = y_scaled.index)
    df_y_unscaled.index.name = 'datetime'
    
    return df_y_unscaled


def prepare_final_dataframe(df, df_raw, data_source, target, model_version, data_version):
    """
    Preparing for the output to be stored in the database. 
    
    df_raw: the raw dataframe containig the target variable
    data_source: the data source for which the predictions are made, e.g. 'resono'
    """

    df_final = df.copy()
    
    if data_source == 'resono':
        
        if target == "count":
            # Set negative predictions to zero
            df_final[df_final < 0] = 0
    
        # Long format
        df_final = pd.melt(df_final.reset_index(), id_vars = 'datetime')
        df_final = df_final.sort_values("datetime", ascending = True)
    
        # Rename columns
        if target == 'count':
            df_final = df_final.rename(columns = {'value': 'total_count_predicted', 'datetime': 'measured_at',
                                                  'variable': 'location_name'})
            # Round to integers
            df_final['total_count_predicted'] = df_final['total_count_predicted'].round()
            
        elif target == "level":
            df_final = df_final.rename(columns = {'value': 'crowd_level_predicted', 'datetime': 'measured_at',
                                                  'variable': 'location_name'})
             # Round to integers
            df_final['crowd_level_predicted'] = df_final['crowd_level_predicted'].round()
    
    # Information on the predictions
    df_final['model_version'] = model_version
    df_final['data_version'] = data_version
    df_final['predicted_at'] = datetime.now()
    
    # Merge with original raw data frame
    df_final = df_raw.merge(df_final, how = 'right')
    
    # Select prediction time slots
    if data_source == "resono":
        df_final = df_final[df_final["measured_at"].isin(df.index)]
    
    return df_final
    


