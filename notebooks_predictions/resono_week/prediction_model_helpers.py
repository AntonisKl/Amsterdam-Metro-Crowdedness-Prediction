# General
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from sqlalchemy import create_engine, inspect
import copy

# Dates
from datetime import datetime, timedelta, date
import datetime as dt
import pytz

# Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sn

# Modelling
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
import mord
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Flatten, Dropout
#from keras.optimizers import Adam
#import tensorflow
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils

# Cleaning
from scipy.optimize import curve_fit
from sklearn import preprocessing, svm

# Sample weights
from sklearn.utils.class_weight import compute_sample_weight

# Synthetic minority oversampling
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE


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


def preprocess_resono_daily_counts_data(df_raw, freq, end_prediction):
    """
    Prepare the raw resono data with daily visitor counts for modelling. 
    """
    
    # Drop duplicates
    df = df_raw.copy()
    df = df.drop_duplicates() 
    
    # Fix timestamp
    df["datetime"] = pd.to_datetime(df["Datum"])
    df = df.sort_values(by = "datetime", ascending = True)
        
    # Wide format
    df = df.pivot_table(index = ["datetime"], columns = "Location", values = "Unique").reset_index()
    df = df.set_index('datetime')
        
    # Change column names
    df.rename(columns = {'Location': 'location'})
    
    # Set right sampling frequency
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  # Any new samples are treated as missing data
    
    return df


def preprocess_resono_hourly_counts_data(df_raw, freq, end_prediction):
    """
    Prepare the raw resono data with daily visitor counts for modelling. 
    """
    
    # Drop duplicates
    df = df_raw.copy()
    df = df.drop_duplicates() 
    
    # Fix timestamp
    df["datetime"] = pd.to_datetime(df['Date'], format='%Y%m%dT%H:%M:%S.%f') + pd.to_timedelta(df["Hour"], 
                                                                                               unit = 'hours')
    df = df.sort_values(by = "datetime", ascending = True)
        
    # Wide format
    df = df.pivot_table(index = ["datetime"], columns = "Location", values = "Visits").reset_index()
    df = df.set_index('datetime')
        
    # Change column names
    df.rename(columns = {'Location': 'location'})
    
    # Set right sampling frequency
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  # Any new samples are treated as missing data
    
    return df


def preprocess_gvb_data(df_raw, freq, end_prediction):
    """
    Prepare the raw GVB data for modelling. 
    """
    
    # create datetime column
    df_raw['date'] = pd.to_datetime(df_raw['Datum'])
    df_raw['time'] = df_raw['UurgroepOmschrijving (van aankomst)'].astype(str).str[:5]
    df_raw['datetime'] = df_raw['date'].astype(str) + " " + df_raw['time']
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        
    # fix amount of travels
    df_raw['AantalReizen'] = pd.to_numeric(df_raw['AantalReizen']) 
    df_raw = df_raw.loc[df_raw['AantalReizen'].notna()]
      
    # drop duplicates
    df = df_raw.drop_duplicates()
    
    # set index
    df = df.set_index('datetime')
    
    # remove columns
    df = df.drop(['Datum', 'UurgroepOmschrijving (van aankomst)', 'date', 'time', 
                  'AankomstLat', 'AankomstLon'], axis = 1)
    
    # rename columns
    df = df.rename(columns = {'AankomstHalteCode': 'location_id',
                              'AankomstHalteNaam': 'location_name', 
                              'AantalReizen': 'total_count'}) 
    
    # wide format
    df = df.pivot_table(index = "datetime", columns = "location_name", values = "total_count").reset_index()
    
    # set index again
    df = df.set_index('datetime')
    df.index.name = 'datetime'
    
    # Set right sampling frequency
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  # Any new samples are treated as missing data
    
    # Fill NaNs from 23:00-05:45 with 0 (no check outs possible)
    idx = df.index.isin(df.between_time('23:00:00', '05:45:00').index)
    df[idx] = 0
    
    # Fill NaNs using most recent value (only for missing values due to upsampling)
    df = df.fillna(method = "ffill", limit = 3)
    
    return df


def preprocess_cmsa_data(df_raw, freq, end_prediction): 
    """
    Prepare the raw CMSA data for modelling. 
    """
 
    df = df_raw.copy()
    
    # get datetime
    df['datetime'] = pd.to_datetime(df['measured_at'], utc = False)
    
    # sort dateframe by timestamp and set index
    df = df.sort_values(by = 'datetime', ascending = True)
    df = df.set_index('datetime')
    
    # Add ID to location names
    df["location_name"] = df["location_name"] + "_cmsa"
    
    # remove columns
    df = df.drop(['neighbourhood', 'neighbourhood_code', 'area',  'district', 'district_code', 
                  'measured_at', 'location_desc', 'geo', 'location_id', 'status'], axis = 1)
    
    # Wide format
    df = df.pivot_table(index = ["datetime"], columns = "location_name", values = "total_count").reset_index()
    df = df.set_index('datetime')
    
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


def preprocess_vacation_data(df_raw, freq, end_prediction):
    """
    Prepare the raw vacation data for modelling. 
    """
    
    # Create datetime index
    df_raw['datetime'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index('datetime')
    
    # Create dummy variable
    df_raw['vacation_dummy'] =  1
    df = df_raw.resample('1d').asfreq()
    df['vacation_dummy'] = df['vacation_dummy'].fillna(0)
    df['vacation_dummy'] = df['vacation_dummy'].astype(int)
    
    # Select column
    df = df[['vacation_dummy']]
    
    # Set right sampling frequency
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  
    
    # Fill missing values with nearest value
    df = df.fillna(method = "ffill")
    
    # Set back to right dtype
    df = df['vacation_dummy'].astype(int)
    
    return df

def preprocess_parking_data(df_raw, freq, end_prediction, remove_ceiling = False):
    """
    Prepare the raw parking data for modelling. 
    
    remove_ceiling: True/False, treat max. capacity samples as missing to impute later (remove ceiling effect).
    """
    
    # Drop duplicates
    df = df_raw.copy()
    df = df.drop_duplicates() 
    
    # Fix timestamp
    df["measured_at"] = pd.to_datetime(df["measured_at"]) 
    df = df.sort_values(by = "measured_at", ascending = True)

    # Change column names
    df = df.rename(columns = {'measured_at': 'datetime', 'location_name': 'location'}) 
    
    # Add column with percentage of short term spots in use 
    df['occupied'] = ((df['short_max_capacity'] - df['short_spots_available']) / df['short_max_capacity']) * 100
    
    # Wide format
    df = df.pivot_table(index = ["datetime"], columns = "location", values = "occupied").reset_index()
    df = df.set_index('datetime')
    
    # Down sample to every 15 minutes (15, 30, 45, 00)
    df = df.resample(freq, label='left', closed='right').mean() 
    
    # Set right sampling frequency
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  
    
    # Set max occupation to missing so that the estimate will become higher when imputing later 
    if remove_ceiling:
        df[df == 100] = float("NaN")
    
    return df


def preprocess_knmi_data(df_raw, freq, end_prediction, resono_hourly_counts = False, resono_daily_counts = False):
    """
    Prepare the raw knmi data consisting of observations for modelling.
    Documentation: https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
    
    resono_daily_counts: set to True if daily historic Resono data is used (sampling rate daily)
    resono_hourly_counts: set to True if daily historic Resono data is used (sampling rate hourly)
    """
    
    # drop duplicates
    df_raw = df_raw.drop_duplicates()    
    
    # rename columns
    df = df_raw.rename(columns={"DD": "wind_direction", "FH": "wind_speed_h", "FF": "wind_speed", "FX": "wind_gust",
                            "T": "temperature", "T10N": "temperature_min", "TD": "dew_point_temperature",
                            "SQ": "radiation_duration", "Q": "global_radiation",
                            "DR": "precipitation_duration", "RH": "precipitation_h",
                            "P": "pressure", "VV": "sight", "N": "cloud_cover", "U": "relative_humidity",
                            "WW": "weather_code", "IX": "weather_index",
                            "M": "fog", "R": "rain", "S": "snow", "O": "thunder", "Y": "ice"
                           })    
    
    # get proper datetime column
    df["date"] = pd.to_datetime(df['date'], format='%Y%m%dT%H:%M:%S.%f')
    if (resono_hourly_counts) | (resono_daily_counts):
        df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit = 'hours')
    else:
        df["datetime"] = df["date"] + pd.to_timedelta(df["hour"] - 1, unit = 'hours')
        df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam")
    df = df.sort_values(by = "datetime", ascending = True)
    df = df.set_index('datetime')  
    
    # Change format datetime
    df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    
    # drop unwanted columns
    df = df.drop(['date', 'hour',
                 'weather_code', 'station_code', 'temperature_min'], 
                 axis = 'columns')   # Many missing values temperature_min  
    
    
    # Feature selection
    df = df[['temperature', 'global_radiation', 'wind_speed', 'cloud_cover']]
    
    # Set right sampling frequency
    if resono_daily_counts:
        df = df.resample(freq).mean()
        
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  # Any new samples are treated as missing data    
    df = df.fillna(method = 'ffill', limit = 3)
    
    return df


def preprocess_metpre_data(df_raw, freq, end_prediction, resono_hourly_counts = False, resono_daily_counts = False):
    """
    Prepare the raw knmi data consisting of predictions for modelling.
    Documentation: https://www.meteoserver.nl/weersverwachting-API.php
    """
    
    # rename columns
    df = df_raw.rename(columns={"windr": "wind_direction", "rv": "relative_humidity", "luchtd": "pressure",
                                "temp": "temperature", "windb": "wind_force",  "winds": "wind_speed",
                                "gust": "wind_gust", "vis": "sight_m", "neersl": "precipitation_h",
                                "gr": "global_radiation", "tw": "clouds"
                           })    
    
    # drop duplicates
    df = df.drop_duplicates()    
    
    # get proper datetime column
    if (resono_hourly_counts) | (resono_daily_counts):
        df["datetime"] = pd.to_datetime(df['tijd'], unit='s')
        df["datetime"] = df["datetime"] + pd.to_timedelta(1, unit = 'hours')  ## klopt dan beter, maar waarom?
    else:
        df["datetime"]  = pd.to_datetime(df['tijd'], unit='s', utc = True)
        df["datetime"] = df["datetime"] + pd.to_timedelta(1, unit = 'hours')  ## klopt dan beter, maar waarom?
        
    df = df.sort_values(by = "datetime", ascending = True) 
    df = df.set_index("datetime")
    
    # Change format datetime
    df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    
    # new column: forecast created on
    df["offset_h"] = df["offset"].astype(float)
    #df["datetime_predicted"] = df["datetime"] - pd.to_timedelta(df["offset_h"], unit = 'hours')  
    
    # select latest prediction 
    # logisch voor prediction set, niet zozeer voor training set
    df = df.reset_index()
    df = df.sort_values(by = ['datetime', 'offset_h'])
    df = df.drop_duplicates(subset = 'datetime', keep = 'first')  
    df = df.set_index("datetime")
    
    # drop unwanted columns
    df = df.drop(['tijd', 'tijd_nl', 'loc',
                  'icoon', 'samenv', 'ico',
                  'cape', 'cond',  'luchtdmmhg', 'luchtdinhg',
                  'windkmh', 'windknp', 'windrltr', 'wind_force',
                  'gustb', 'gustkt', 'gustkmh', 'wind_gust',  
                  # deze zitten er niet in voor 14 juni
                  'hw', 'mw', 'lw',
                  'offset', 'offset_h',
                  'gr_w'], axis = 'columns', errors = 'ignore')    
    
    # set datatypes of weather data to float
    df = df.astype('float64')   
    
    # cloud cover similar to observations (0-9) & sight, but not really the same thing
    df['cloud_cover'] =  df['clouds'] / 12.5
    df['sight'] =  df['sight_m'] / 333
    df.drop(['clouds', 'sight_m'], axis = 'columns')  
    
    # Feature selection
    df = df[['temperature', 'global_radiation', 'wind_speed', 'cloud_cover']]

    # Set right sampling frequency
    if resono_daily_counts:
        df = df.resample(freq).mean()
    elif (not resono_hourly_counts) & (not resono_daily_counts):
        # go from hourly to quarterly values
        df = df.resample(freq).ffill(limit = 11)
        
    idx = pd.date_range(df.index[0], end_prediction, freq = freq)
    df = df.reindex(idx)  # Any new samples are treated as missing data    
    df = df.fillna(method = 'ffill', limit = 3)

    # later misschien smoothen? lijkt nu niet te helpen voor voorspelling
    #df = df.apply(lambda x: savgol_filter(x,17,2))
    
    return df  


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


### FUNCTIONS - cleaning
# put all your cleaning functions here
    
def clean_data(df, target, Y_name, n_samples_day, cols_to_clean, outlier_removal, nu = 0.2, gamma = 0.1, 
               resono_hourly_counts = None, resono_daily_counts = None):
    """
    Clean data by imputing missing values and removing outliers (replacing with interpolation/extrapolation).
    Days that are fully missing are dropped from the dataframe (to prevent strange interpolation results).
    
    cols_to_clean: column names of columns to clean
    outlier_removal: "yes" or "no"
    nu: value for nu parameter for outlier removal model (default is 0.2) 
    gamma: value for gamma parameter for outlier removal model (default = 0.1)
    resono_daily_counts: true if daily week-ahead predictions 
    resono_hourly_counts: true if hourly week-ahead predictions
    
    
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
    
    if (resono_daily_counts is None) & (resono_hourly_counts is None):
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
    
    if (resono_daily_counts is None) & (resono_hourly_counts is None):
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


def add_time_variables(df, daily = None):
    """
    Create dummy variables from weekday, weekend and hour and add these to the dataframe. 
    Also, add cos and sine times
    """
    
    df = df.reset_index()
    
    # add weekday and hour dummies
    df_temp = copy.deepcopy(df)
    df_temp['weekday'] = pd.Categorical(df_temp['datetime'].dt.weekday)
    weekday_dummies = pd.get_dummies(df_temp[['weekday']], prefix='weekday_')
    df_time = df.merge(weekday_dummies, left_index = True, right_index = True)
        
    if not daily:
        df_temp['hour'] =  pd.Categorical(df_temp['datetime'].dt.hour)
        hour_dummies = pd.get_dummies(df_temp[['hour']], prefix='hour_')
 
        df_time = df_time.merge(hour_dummies, left_index = True, right_index = True)
    
        # add cyclical time features
        df_temp['minutes'] = df_temp['datetime'].dt.hour * 60 + df_temp['datetime'].dt.minute
        df_time['sin_time'] = np.sin(2 * np.pi * df_temp['minutes'] / (24 * 60)) 
        df_time['cos_time'] = np.cos(2 * np.pi * df_temp['minutes'] / (24 * 60)) 
        
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


def add_lag_variables_week_ahead(df, Y_name, target, predict_period, n_samples_day, n_samples_week):
    """
    Add lag variables (features that are lagged version of the target variable).
    """

    # Value of previous week(s) and average of those previous values
    df[Y_name + "_prev_week2"] = df[Y_name].shift(n_samples_week*2)
    df[Y_name + "_prev_week3"] = df[Y_name].shift(n_samples_week*3)
    df[Y_name + "_prev_week4"] = df[Y_name].shift(n_samples_week*4)
    df[Y_name + "_prev_weeks_avg"] = df[[Y_name + "_prev_week2", Y_name + "_prev_week3", 
                                         Y_name + "_prev_week4"]].mean(axis = 1)

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
# put all your modelling functions here

def train_model_xxx(df_X_train, df_y_train):     
    
    """
    Train ... model
    """
    #
    #
    #
    return model


def test_model_xxx(model, df_X_test):
    """
    Run trained ... model on the prediction or test set
    """
    #
    #
    #
    return prediction


def test_model_random_walk(df_y_train, predict_period):
    """
    Run random walk (naïve) model on the prediction or test set
    """

    # Use most recent value as predicted values
    prediction = [df_y_train.iloc[-1, 0]] * predict_period
    
    return prediction


def test_model_avg_3_weeks(df_y_train, df_y_predict, predict_period, n_samples_week, target):
    """
    Run model that uses the average of the previous 3 weeks on the prediction or test set
    """

    # Use average of last 3 weeks (for same time stamps) as predicted values
    df_hist = pd.concat([df_y_train, df_y_predict], 0)
    df_hist_1week = df_hist.shift(n_samples_week)
    df_hist_2week = df_hist.shift(2*n_samples_week)
    df_hist_3week = df_hist.shift(3*n_samples_week)
    df_hist_all = pd.concat([df_hist_1week, df_hist_2week, df_hist_3week], 1)
    df_hist_all = df_hist_all[df_hist_all.index.isin(df_y_predict.index)]
    
    if target == "count":
        # Average
        prediction = df_hist_all.mean(axis = 1)
    elif target == "level":
        # Majority class
        prediction = df_hist_all.mode(axis = 1)
    
    return prediction


def test_model_past_week(df_y_train, df_y_predict, predict_period, n_samples_week, target):
    """
    Run model that uses the past week on the prediction or test set
    """

    # Use past week(for same time stamps) as predicted values
    df_hist = pd.concat([df_y_train, df_y_predict], 0)
    df_hist_1week = df_hist.shift(n_samples_week)
    df_hist_1week = df_hist_1week[df_hist_1week.index.isin(df_y_predict.index)]
    
    prediction = df_hist_1week.copy()
    
    return prediction


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


def train_model_ordinal_regression(df_X_train, df_y_train, Y_name, target, use_smote = False):     
    
    """
    Train ordinal regression model
    
    use_smote: True/False (synthetic minority oversampling)
    """
    
    # Perform synthetic minority oversampling
    if use_smote:
        X_train, y_train = perform_smote(df_X_train, df_y_train, Y_name, target, thresholds)
        
        # Convert y_train to integer dtype (necessary for this model)
        y_train = y_train.astype(int)
        
    else:
        # Convert data to numpy array
        X_train = np.array(df_X_train)
        y_train = np.array(df_y_train)
        y_train = y_train.reshape(len(y_train))
        
        # Convert y_train to integer dtype (necessary for this model)
        y_train = y_train.astype(int)
    
    # Initialize model (all threshold based model)
    model = mord.LogisticAT()
              
    # Fit model
    if use_sample_weights:
        model.fit(X_train, y_train, sample_weight = sample_weights)
    else:
        model.fit(X_train, y_train)
    
    return model


def test_model_ordinal_regression(model, df_X_test):
    """
    Run trained ordinal regression model on the prediction or test set
    """

    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(X_test)
    
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


def train_model_LSTM_regression(df_X_train, df_y_train, prediction_window,
                               tune_hyperparameters, batch_size, epochs, neurons, drop_out_perc, learning_rate):
    """
    Train LSTM regression model.
    
    tune_hyperparameters: True/False (perform grid search to tune hyperparameters)
    
    List (tune_hyperparameters = True) or single values (tune_hyperparameters = False) of value(s) for:
    batch_size: number of samples shown to the model at one iteration 
    epochs: number of epochs used to train the model
    neurons: number of neurons in the LSTM layer
    drop_out_perc: percentage of neurons used in the drop out layers
    learning_rate: learning rate used for weight updates
    """
    
    # Crop training data so that the number of samples is divisible by the prediction window
    new_length = int(prediction_window * math.floor(len(df_y_train)/prediction_window))
    
    df_y_train = df_y_train[len(df_y_train)-new_length:]
    df_X_train = df_X_train.iloc[len(df_X_train)-new_length:, :]
    
    # Number of features
    n_features = len(df_X_train.columns)
    
    # Convert data to numpy array
    X_train = np.array(df_X_train)
    y_train = np.array(df_y_train)

    # Reshape predictor variables into 3D array
    X_train = X_train.reshape(int(len(df_X_train) / prediction_window), prediction_window, n_features)
    
    # Reshape target variable into 2D array
    y_train = y_train.reshape(int(len(y_train) / prediction_window), prediction_window)
        
    if tune_hyperparameters:
        
        # Single values also have to be in a list
        prediction_window = [prediction_window]
        n_features = [n_features]
        
        # Determine optimal hyperparameters based on training data 
        opt_hp = hyperparameter_search_LSTM_regression(X_train, y_train, batch_size, epochs, neurons, drop_out_perc,
                                                       learning_rate, prediction_window, n_features)
    
        # Store optimal hyperparameters
        hyperparameters = opt_hp
    
        # Select optimal hyperparameters
        drop_out_perc = opt_hp['drop_out_perc']
        learning_rate = opt_hp['learning_rate']
        neurons = opt_hp['neurons']
        batch_size = opt_hp['batch_size']
        epochs = opt_hp['epochs']
        prediction_window = opt_hp['prediction_window']
        n_features = opt_hp['n_features']
        
    # Create sequential model
    model = Sequential()
                    
    # Add dropout layer
    model.add(Dropout(drop_out_perc, input_shape=(prediction_window, n_features)))
    
    # Add dense layer
    model.add(LSTM(neurons, activation='relu', return_sequences = True)) 
            
    # Add dropout layer
    model.add(Dropout(drop_out_perc))
                
    # Add an output layer with one unit
    model.add(Dense(1))
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    
    # Compile the model 
    model.compile(optimizer=optimizer, loss='mse')

    # Fit the model with the selected hyperparameters
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
    
    if tune_hyperparameters:
        return model, hyperparameters
    else:
        return model
                   
    
def initialize_LSTM_regression(batch_size, epochs, neurons, drop_out_perc, learning_rate, prediction_window, n_features):
    """
    Initialize the LSTM model for regression
    """
    
    # Create sequential model
    init_model = Sequential()
    
    # Add dropout layer
    init_model.add(Dropout(drop_out_perc, input_shape=(prediction_window, n_features)))
    
    # Add LSTM layer
    init_model.add(LSTM(neurons, activation='relu', return_sequences = True))
    
    # Add dropout layer
    init_model.add(Dropout(drop_out_perc))
     
    # Add an output layer with one unit
    init_model.add(Dense(1))
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    
    # Compile the model 
    init_model.compile(optimizer=optimizer, loss='mse')
    
    return init_model


def test_model_LSTM_regression(model, df_X_test):
    """
    Run trained LSTM regression model on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Reshape predictor variables into 3D array
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    
    # Predict new data 
    prediction = model.predict(X_test).flatten()

    return prediction

        
def hyperparameter_search_LSTM_regression(X_train, y_train, batch_size, epochs, neurons, drop_out_perc, 
                                          learning_rate, prediction_window, n_features):
    """ 
    Perform a hyperparameter grid search on the training data (only for LSTM regression model)
    """
    
    # Initialize LSTM
    init_model = KerasClassifier(build_fn=initialize_LSTM_regression, verbose=0)

    # Initialize a dictionary with the hyper parameter options
    param_grid = dict(batch_size = batch_size, epochs = epochs, neurons = neurons,
                     drop_out_perc = drop_out_perc, learning_rate = learning_rate, prediction_window = prediction_window,
                      n_features = n_features)

    # Train the model and find the optimal hyper parameter settings
    grid_result = GridSearchCV(estimator=init_model, param_grid=param_grid, n_jobs=-1, cv=3, 
                               scoring = 'neg_mean_squared_error', refit = True)
 
    # Fit the model to retrieve the best parameters
    grid_result.fit(X_train, y_train)
        
    # Select the best hyper parameters 
    opt_hp = grid_result.best_params_
                
    return opt_hp

    
def train_model_LSTM_classification(df_X_train, df_y_train, prediction_window, batch_size, epochs, neurons, drop_out_perc,
                                   learning_rate):
    """
    Train LSTM classification model.
    
    prediction_window: the prediction window in terms of samples/time steps (equal to predict_period)
    batch_size: number of samples shown to the model at one iteration 
    epochs: number of epochs used to train the model
    neurons: number of neurons in the LSTM layer
    drop_out_perc: percentage of neurons used in the drop out layers
    learning_rate: learning rate used for updating the weights
    """   
    
    # Crop training data so that the number of samples is divisible by the prediction window
    new_length = int(prediction_window * math.floor(len(df_y_train)/prediction_window))
    
    df_y_train = df_y_train[len(df_y_train)-new_length:]
    df_X_train = df_X_train.iloc[len(df_X_train)-new_length:, :]
    
    # Number of features
    n_features = len(df_X_train.columns)
    
    # Convert data to numpy array
    X_train = np.array(df_X_train)
    y_train = np.array(df_y_train)
    
    # Number of classes
    n_classes = len(np.unique(y_train))

    # Reshape predictor variables into 3D array
    X_train = X_train.reshape(int(len(X_train) / prediction_window), prediction_window, n_features)
    
    # Reshape target variable into 2D array
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    y_train = np_utils.to_categorical(encoded_y_train)
    y_train = y_train.reshape(int(len(y_train) / prediction_window), prediction_window, n_classes)
                                  
    # Create sequential model
    model = Sequential()
                    
    # Add dropout layer
    model.add(Dropout(drop_out_perc, input_shape=(prediction_window, n_features)))
    
    # Add dense layer
    model.add(LSTM(neurons, activation='relu', return_sequences = True)) 
            
    # Add dropout layer
    model.add(Dropout(drop_out_perc))
                
    # Add an output layer with as many units as classes
    model.add(Dense(n_classes, activation = "softmax"))
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    
    # Compile the model 
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # Fit the model with the chosen hyperparameters
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
                                               
    return model


def test_model_LSTM_classification(model, df_X_test):
    """
    Run trained LSTM classification model on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Reshape predictor variables into 3D array
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    
    # Predict data 
    prediction = model.predict(X_test)
         
    # Reshape predictions
    prediction = prediction.reshape(prediction.shape[1], prediction.shape[2])
                
    # Predict class with highest probability
    prediction = np.argmax(prediction, axis = 1)
    prediction = np.where(prediction == 0, -1.0, np.where(prediction == 1, 0.0, 1.0))
    
    return prediction


def train_model_RF_regression(df_X_train, df_y_train):     
    
    """
    Train Random Forest regression model
    """
    
    # Convert data to numpy array
    X_train = np.array(df_X_train)
    y_train = np.array(df_y_train)
    
    # Reshape target variable into 1D array
    y_train = y_train.reshape(len(y_train))
    
    # Initialize model
    model = RandomForestRegressor()
      
    # Fit model
    model.fit(X_train, y_train)
    
    return model


def test_model_RF_regression(model, df_X_test):
    """
    Run trained Random Forest regression model on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(df_X_test)
    
    return prediction


def train_model_RF_classification(df_X_train, df_y_train, use_sample_weights = False, use_smote = False):     
    
    """
    Train Random Forest classification model
    
    use_sample_weights: True/False
    use_smote: True/False (synthetic minority oversampling)
    """
    
    # Perform synthetic minority oversampling
    if use_smote:
        X_train, y_train = perform_smote(df_X_train, df_y_train)
        
    else:
        # Convert data to numpy array
        X_train = np.array(df_X_train)
        y_train = np.array(df_y_train)
        y_train = y_train.reshape(len(y_train))
    
    # Calculate sample weights
    if use_sample_weights:
        sample_weights = compute_sample_weight('balanced', y_train)
        
    # Initialize model (all threshold based model)
    model = RandomForestClassifier()
              
    # Fit model
    if use_sample_weights:
        model.fit(X_train, y_train, sample_weight = sample_weights)
    else:
        model.fit(X_train, y_train)
    
    return model


def test_model_RF_classification(model, df_X_test):
    """
    Run trained Random Forest classification model on the prediction or test set
    """

    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(X_test)
    
    return prediction


def train_model_xg(df_X_train, df_y_train):     
    
    """
    Train XGBoost model
    """
    
    # Convert data to numpy array
    X_train = np.array(df_X_train)
    y_train = np.array(df_y_train)
    
    # Reshape target variable into 1D array
    y_train = y_train.reshape(len(y_train))

    # Initialize model
    #hyper_parameters = {
              #'learning_rate': [0.1],
              #'max_depth': [6],
              #'colsample_bytree': [0.2],
              #'n_estimators': [70],
              #'alpha': [9]}
                
    model =  xgb.XGBRegressor(objective ='reg:squarederror', 
                              colsample_bytree = 0.2, 
                              learning_rate = 0.1,
                              max_depth = 6,  # perhaps higher? 
                              alpha = 9, 
                              n_estimators = 70) # perhaps higher?
    
    #objective ='reg:squarederror', 
                              #colsample_bytree = 0.3, 
                              #learning_rate = 0.1,
                              #max_depth = 3, 
                              #alpha = 20, 
                              #n_estimators = 40
                        
    #model_grid = GridSearchCV(model,
                        #hyper_parameters,
                        #cv = 5,
                        #n_jobs = -1,
                        #scoring = 'neg_mean_squared_error',
                        #verbose=True)
    
    # Fit model
    model.fit(df_X_train, df_y_train)
    
    # Root Mean Squared Error
    #print(np.sqrt(-model_grid.best_score_))
    #print(model_grid.best_params_)
    
    return model


def test_model_xg(model, df_X_test):
    """
    Run trained XGBoost model on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # basic prediction
    prediction = model.predict(df_X_test)
       
    return prediction
    
    
### FUNCTIONS - post operational modelling
# put all your post-processing functions here

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


def prepare_final_dataframe(df, df_raw, data_source, target, model_version, data_version, resono_daily_counts = None,
                           resono_hourly_counts = None):
    """
    Preparing for the output to be stored in the database. 
    
    df_raw: the raw dataframe containing the target variable
    data_source: the data source for which the predictions are made, e.g. 'resono'
    resono_daily_counts: True if using daily historic counts 
    resono_hourly_counts: True if usng hourly historic counts
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
            if resono_daily_counts:
                df_final = df_final.rename(columns = {'value': 'total_count_predicted', 'datetime': 'Datum',
                                                      'variable': 'Location'})
                df_raw['Datum'] = pd.to_datetime(df_raw['Datum'])  
            elif resono_hourly_counts:
                df_final = df_final.rename(columns = {'value': 'total_count_predicted', 'variable': 'Location'})
                df_raw["datetime"] = pd.to_datetime(df_raw['Date'], 
                                                    format='%Y%m%dT%H:%M:%S.%f') + pd.to_timedelta(df_raw["Hour"], 
                                                                                               unit = 'hours')
                df_raw = df_raw.sort_values(by = "datetime", ascending = True)
            else:
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
        if resono_daily_counts:
            df_final = df_final[df_final["Datum"].isin(df.index)]
        elif resono_hourly_counts:
            df_final = df_final[df_final["datetime"].isin(df.index)]
        else:    
            df_final = df_final[df_final["measured_at"].isin(df.index)]
    
    return df_final


### FUNCTIONS - backtesting 
# put all your backtesting functions here
# these prediction functions should use the train_model and test_model functions from above as well

def prepare_backtesting(start_test, predict_period, freq, df, Y_name, n_samples_week, target,
                       y_scaler):
    '''
    Prepare the data frames for backtesting given the starting timestamp for the predictions.
    '''
    
    # Define end timestamp of predictions
    end_test = pd.date_range(start_test, periods = predict_period, freq = freq)
    end_test = end_test[len(end_test)-1]
    
    # Training data
    df_X_train_bt, df_y_train_bt = get_train_df(df, Y_name, start_test)
    
    # Data frame to fill in with predictions
    df_y_predict_bt = get_future_df(start_test, predict_period, freq)
    
    # Drop ground truth values
    df_X_predict_bt = df.drop(Y_name, 1)

    # Select features for prediction period
    df_X_predict_bt = df_X_predict_bt[start_test:end_test]
    
    # Save ground truth values
    df_y_ground_truth_bt = df[[Y_name]][start_test:end_test]
    
    # If days are missing from ground truth, also discard them in data frame to predict
    df_y_predict_bt = df_y_predict_bt[df_y_predict_bt.index.isin(df_y_ground_truth_bt.index)]
    
    # Save scaled ground truth values
    df_y_ground_truth_bt_scaled = df_y_ground_truth_bt
    
    # Unscale the ground truth data if continuous
    if target == "count":
        df_y_ground_truth_bt = unscale_y(df_y_ground_truth_bt, y_scaler)
        
    return df_y_predict_bt, df_y_train_bt, df_y_ground_truth_bt, df_y_ground_truth_bt_scaled, df_X_train_bt, df_X_predict_bt


def test_model_avg_3_weeks_bt(df_y_train, df_y_predict, df_y_ground_truth_scaled, predict_period, n_samples_week, target):
    """
    Run model that uses the average of the previous 3 weeks on the prediction or test set
    """

    # Use average of last 3 weeks (for same time stamps) as predicted values
    df_hist = pd.concat([df_y_train, df_y_ground_truth_scaled], 0)
    df_hist_1week = df_hist.shift(n_samples_week)
    df_hist_2week = df_hist.shift(2*n_samples_week)
    df_hist_3week = df_hist.shift(3*n_samples_week)
    df_hist_all = pd.concat([df_hist_1week, df_hist_2week, df_hist_3week], 1)
    df_hist_all = df_hist_all[df_hist_all.index.isin(df_y_predict.index)]
    
    if target == "count":
        # Average
        prediction = df_hist_all.mean(axis = 1)
    elif target == "level":
        # Majority class
        prediction = df_hist_all.mode(axis = 1).iloc[:, 0]
    
    return prediction


def test_model_past_week_bt(df_y_train, df_y_predict, df_y_ground_truth_scaled, predict_period, n_samples_week, target):
    """
    Run model that uses the past week on the prediction or test set
    """

    # Use past week(for same time stamps) as predicted values
    df_hist = pd.concat([df_y_train, df_y_ground_truth_scaled], 0)
    df_hist_1week = df_hist.shift(n_samples_week)
    df_hist_1week = df_hist_1week[df_hist_1week.index.isin(df_y_predict.index)]
    
    prediction = df_hist_1week.copy()
    
    return prediction


def evaluate(pred, ground_truth, target, count_to_level = False, Y_name = None, thresholds = None, print_metrics=True):
    """
    Evaluate a prediction and ground_truth.
    
    count_to_level: convert results to crowd level and show confusion matrix using the given thresholds
    """
    
    if target == 'count':
        
        # fill NaNs with zeroes
        pred = pred.fillna(method = "ffill")
        pred = pred.fillna(method = "bfill")
        ground_truth = ground_truth.fillna(method = "ffill")
        ground_truth = ground_truth.fillna(method = "bfill")
        
        # Set negative predictions to zero
        pred[pred < 0] = 0
        ground_truth[ground_truth < 0] = 0
    
        # Calculate error metrics
        rmse = mean_squared_error(ground_truth, pred, squared=False)
        mae = mean_absolute_error(ground_truth, pred)
      
        # Calculate error metrics only for crowded moments (p75)   
        busy = np.percentile(ground_truth, 75)
        ground_truth_busy = ground_truth[ground_truth > busy].dropna()
        pred_busy = pred[ground_truth > busy].dropna()
        rmse_busy = mean_squared_error(ground_truth_busy, pred_busy, squared=False)
        mae_busy = mean_absolute_error(ground_truth_busy, pred_busy)
        
        # Store error metrics in dict
        error_metrics = dict({'rmse': rmse, 'rmse_busy': rmse_busy, 'mae': mae, 'mae_busy': mae_busy})
    
        if print_metrics:
            print(f"Root mean squared error: {rmse.round(1)}")
            print(f"Root mean squared error (crowded): {rmse_busy.round(1)}")
            print(f"Mean absolute error: {mae.round(1)}")
            print(f"Mean absolute error (crowded): {mae_busy.round(1)}")
            
        if count_to_level:
            pred = get_crowd_levels(pred, Y_name, thresholds)
            ground_truth = get_crowd_levels(ground_truth, Y_name, thresholds)
            
            # Confusion matrix
            conf_mat = confusion_matrix(ground_truth, pred)
            
            error_metrics['conf_mat'] = conf_mat
            
    elif target == "level":
        
        # Set dtype to category
        pred = pred.astype('category')
        ground_truth = ground_truth.astype('category')
        
        # Forward fill NaNs
        pred = pred.fillna(method = "ffill")
        ground_truth = ground_truth.fillna(method = "ffill")
        
        # Confusion matrix
        conf_mat = confusion_matrix(ground_truth, pred)
        
        # Classification report (recall, precision, F1)
        class_report = classification_report(ground_truth, pred, output_dict = True)
        class_report = pd.DataFrame(class_report).transpose()
        
        error_metrics = dict({"conf_mat": conf_mat, "class_report": class_report})
        
        if print_metrics:
            print(f"Confusion matrix: {conf_mat}")
            print(f"Classification report: {class_report}")
            
    return error_metrics
        
    
def visualize_backtesting(df_y_ground_truth_bt, df_y_benchmark, df_y_model, target, Y_name, error_metrics, y_label,
                          count_to_level = False):
    '''
    Count: Plot the ground truth, benchmark and model predictions in one graph. 
    Optional: Plot the confusion matrix for the model predictions.
    
    Level: Plot the confusion matrix for the model predictions.
    '''

    if target == "count":
        fig, ax = plt.subplots(figsize = (30, 5))
        df_y_ground_truth_bt.plot(ax = ax)
        df_y_benchmark.plot(ax = ax)
        df_y_model.plot(ax = ax)
        plt.legend(["Ground truth", "Benchmark", "Model"])
        plt.ylabel(y_label)
        plt.xlabel("Date")
        
        plt.close()
        
        if count_to_level:
            sn.set(font_scale=1.5)
            fig2 = sn.heatmap(error_metrics['conf_mat'], 
                         annot=True, cbar = False, fmt='g', cmap = 'Blues').get_figure()
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.xticks(rotation = 45)
            
            plt.close()
            
            return fig, fig2

    elif target == "level":
        sn.set(font_scale=1.5)
        fig = sn.heatmap(error_metrics['conf_mat'], 
                         annot=True, cbar = False, fmt='g', cmap = 'Blues').get_figure()
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.xticks(rotation = 45)
    
    return fig 
    
    
def feature_importance(weights, feature_names):
    ''' 
    Plot the feature weights
    
    feat_imp: array of weights/feature importances of the fitted model 
           (e.g. model.coef_ for regression model, model.feature_importances_ for RF model)
    feature_names: list of names for the features
    '''
     
    # Dataframe with feature weights/importance
    feat_imp = pd.DataFrame(weights, index = feature_names, 
                            columns =["Importance"]).sort_values("Importance")
    feat_imp['Sign'] = np.where(feat_imp <= 0, 'neg', 'pos')
    feat_imp['Feature'] = feat_imp.index
    feat_imp = pd.concat([feat_imp.head(5), feat_imp.tail(5)], 0)
    
    # Create figure of feature weights/importance
    fig = px.bar(feat_imp, x = "Importance",  y = 'Feature', labels={'value':'Importance', 'index':'Feature'}, 
             color = 'Sign', color_discrete_map={'neg':'red', 'pos':'blue'}, orientation = 'h')
    fig.update_layout(showlegend=False)
    
    return feat_imp, fig

def backtesting_results_all_locations(locations, RMSE_models, RMSE_benchmarks):
    ''' 
    Create a dataframe with the backtesting results summarized for all locations.
    '''
    
    df_results = pd.DataFrame()
    df_results["Location"] = locations
    df_results["RMSE_model"] = RMSE_models
    df_results["RMSE_benchmark"] = RMSE_benchmarks
    df_results["RMSE_difference"] = np.subtract(df_results["RMSE_model"], df_results["RMSE_benchmark"])
    
    return df_results
    


