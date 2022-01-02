import pandas as pd
import geopandas as gpd
import numpy as np
import os
from sqlalchemy import create_engine

import env

from datetime import datetime, timedelta, date
import pytz

from pathlib import Path
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error 
from sklearn import linear_model
from scipy.signal import savgol_filter

import xgboost as xgb


### FUNCTIONS - pre modelling

def preprocess_cmsa_data(df_raw, my_locations, start_ds): 
    """
    Prepare the raw cmsa data for modelling. 
    We select the sensor locations which are going to be modelled. 
    Also, we will create a proper timestamp. 
    """
    
    # adjust location names if changed
    df_raw['location_id'] = df_raw['location_id'].str.replace('GKS-01-Kalverstraat', 'CMSA-GAKH-01')

    # select locations
    df = df_raw[df_raw["location_id"].isin(my_locations)]
    
    # get datetime
    df["datetime_utc"] = pd.to_datetime(df["measured_at"], utc = True)
    df["datetime"] = df["datetime_utc"].dt.tz_convert("Europe/Amsterdam")
    del df["datetime_utc"]
    del df["measured_at"]
    
    # shift two hours since July 11
    df['datetime'][df['datetime'] > '2021-07-11 15:15:00+02:00'] = df["datetime"][df['datetime'] > '2021-07-11 15:15:00+02:00'] - pd.to_timedelta(2, unit = 'hours')
    df = df.drop_duplicates()
        
    # sort dateframe by timestamp 
    df = df.sort_values(by = "datetime", ascending = True)

    # select only data after starting datetime and reset index
    df = df[df['datetime'] >= start_ds]  ## move this to query later
    df = df.reset_index(drop = True)

    return df


def preprocess_covid_data_api(df_raw):
    """
    Prepare the raw covid stringency data for modelling. 
    """
    
    # Put data to dataframe
    df_raw_unpack = df_raw.T['NLD'].dropna()
    df = pd.DataFrame.from_records(df_raw_unpack) 
    
    # Add datetime column
    df['datetime'] = pd.to_datetime(df['date_value']).dt.tz_localize("Europe/Amsterdam", ambiguous = "NaT")
    
    # Select columns
    df_sel = df[['datetime', 'stringency', 'stringency_legacy']]
        
    # extend dataframe to future (based on latest value
    dates_future = pd.date_range(df['datetime'].iloc[-1], periods = 14, freq='1d')
    df_future = pd.DataFrame(data = {'datetime': dates_future,
                                     'stringency': df['stringency'].iloc[-1],
                                     'stringency_legacy': df['stringency_legacy'].iloc[-1]})
    
    # Add together and set index
    df_final = df_sel.append(df_future.iloc[1:])
    df_final = df_final.set_index('datetime')
    
    # Add covid index shops
    df_final['shopping_restricted'] = 0
    df_final['shopping_restricted'][(df_final.index > '2020-12-14 00:00') & (df_final.index < '2021-04-28 00:00')] = 1

    return df_final


def preprocess_knmi_data(df_raw, start_ds):
    """
    Prepare the raw knmi observations data for modelling. 
    We rename columns and resample from 60min to 15min data.
    Also, we will create a proper timestamp. 
    
    Documentation: https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
    """
    # drop duplicates 
    df_raw = df_raw.drop_duplicates() 

    # rename columns
    df = df_raw.rename(columns={"DD": "wind_direction", "FH": "wind_speed", "FF": "wind_speed_10m", "FX": "wind_gust",
                                "T": "temperature", "T10N": "temperature_min", "TD": "dew_point_temperature",
                                "SQ": "radiation_duration", "Q": "global_radiation",
                                "DR": "precipitation_duration", "RH": "precipitation_h",
                                "P": "pressure", "VV": "sight", "N": "cloud_cover", "U": "relative_humidity",
                                "WW": "weather_code", "IX": "weather_index",
                                "M": "fog", "R": "rain", "S": "snow", "O": "thunder", "Y": "ice"
                               })
    
    # divide some columns by ten (because using 0.1 degrees C etc. as units)
    col10 = ["wind_speed", "wind_speed_10m", "wind_gust", "temperature", "temperature_min", "dew_point_temperature",
             "radiation_duration", "precipitation_duration", "precipitation_h", "pressure"]
    df[col10] = df[col10] / 10
    
    # get proper datetime column
    df["date"] = pd.to_datetime(df['date'], format='%Y%m%dT%H:%M:%S.%f')  
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"] - 1, unit = 'hours')
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam")
    df = df.sort_values(by = "datetime", ascending = True)
    df = df.reset_index(drop = True)
       
    # drop unwanted columns
    df = df[['datetime', 'global_radiation', 'pressure', 'precipitation_h',
             'relative_humidity', 'temperature', 'cloud_cover', 'sight', 
             'wind_direction', 'wind_speed']]
    
    # select only data after starting datetime
    df = df[df['datetime'] >= start_ds] 
    
    # go from hourly to quarterly values
    df_15 = df.set_index('datetime').resample('15min').ffill(limit = 3).reset_index()

    return df_15


def preprocess_metpre_data(df_raw, start_ds):
    """
    Prepare the raw meteoserver predictions data for modelling. 
    We rename and select columns and change order of magnitude for some variables. 
    Also, we create a proper timestamp and select which prediction to use. 
    Finally, we resample from 60min to 15min data.
    Data frame should now be in line with the weather observations. 
    
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
    df["datetime"]  = pd.to_datetime(df['tijd'], unit='s', utc = True)
    df["datetime"] = df["datetime"] + pd.to_timedelta(1, unit = 'hours')  ## aligns better, but why?
    df = df.sort_values(by = "datetime", ascending = True)
    df = df.reset_index(drop = True)
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam")
    
    # new column: forecast created on
    df["offset_h"] = df["offset"].astype(float)
    #df["datetime_predicted"] = df["datetime"] - pd.to_timedelta(df["offset_h"], unit = 'hours')
    
    # select only data after starting datetime
    df = df[df['datetime'] >= start_ds]  
    
    # select latest prediction  ## makes sense for prediction set, not so much for training set 
    df = df.sort_values(by = ['datetime', 'offset_h'])
    df = df.drop_duplicates(subset = 'datetime', keep = 'first')
    
    # drop unwanted columns
    df = df.drop(['tijd', 'tijd_nl', 'loc',
                  'icoon', 'samenv', 'ico',
                  'cape', 'cond',  'luchtdmmhg', 'luchtdinhg',  
                  'windkmh', 'windknp', 'windrltr', 'wind_force',
                  'gustb', 'gustkt', 'gustkmh', 'wind_gust',  ## missing these columns before June 14             
                  'hw', 'mw', 'lw',
                  'offset', 'offset_h',
                  'gr_w'], axis = 'columns', errors = 'ignore')
    
    # set datatypes of weather data to float
    df = df.set_index('datetime')
    df = df.astype('float64').reset_index()
    
    # make cloud cover and sight similar to observations (but not really the same thing)
    df['cloud_cover'] =  df['clouds'] / 12.5
    df['sight'] =  df['sight_m'] / 333
    df.drop(['clouds', 'sight_m'], axis = 'columns')
              
    # go from hourly to quarterly values
    df_15 = df.set_index('datetime').resample('15min').ffill(limit = 11)
    
    # smooth weather predictions  ## smoothing doesn't seem to improve CMSA predictions
    #df_smooth = df_15.apply(lambda x: savgol_filter(x,17,2))
    #df_smooth = df_smooth.reset_index() 
    df_15 = df_15.reset_index()

    return df_15  # df_smooth


def preprocess_weather_data(df_obs, df_pred): 
   
    """
    Merge weather observations and predictions into one continous dataframe
    """
    # Preprocessing of separate dataframes
    # ...
    
    # Select timeframes for different sources
    df_obs = df_obs[df_obs['datetime'] < '2021-06-15 00:00:00+02:00'] 
    df_pred = df_pred[df_pred['datetime'] >= '2021-06-15 00:00:00+02:00']
    
    # Merge all KNMI dataframes 
    df = pd.concat([df_obs, df_pred], ignore_index=True) 
    
    return df


def preprocess_holidays_data(holidays):
    """
    Prepare the raw holiday data for modelling. 
    """
    # Put in dataframe
    holiday_df = pd.DataFrame(holidays).rename(columns = {0: 'date', 1: 'holiday'})
    
    # Create datetime index
    holiday_df['datetime'] = pd.to_datetime(holiday_df['date']).dt.tz_localize("Europe/Amsterdam")
    holiday_df = holiday_df.set_index('datetime')
    
    # Create dummy variable
    holiday_df['holiday_dummy'] =  1
    holiday_df_d = holiday_df.resample('1d').asfreq()
    holiday_df_d['holiday_dummy'] = holiday_df_d['holiday_dummy'].fillna(0)
    holiday_df_d['holiday_dummy'] = holiday_df_d['holiday_dummy'].astype(int)
    
    # Select column
    holiday_df_d = holiday_df_d[['holiday_dummy']]
    
    return holiday_df_d


def preprocess_vacation_data(df_raw):
    """
    Prepare the raw vacation data for modelling. 
    """
    # Create datetime index
    df_raw['datetime'] = pd.to_datetime(df_raw['date']).dt.tz_localize("Europe/Amsterdam")
    df_raw = df_raw.set_index('datetime')
    
    # Create dummy variable
    df_raw['vacation_dummy'] =  1
    df = df_raw.resample('1d').asfreq()
    df['vacation_dummy'] = df['vacation_dummy'].fillna(0)
    df['vacation_dummy'] = df['vacation_dummy'].astype(int)
    
    # Select column
    df = df[['vacation_dummy']]
    
    return df

        
def clean_cmsa_data(cmsa_df):
    
    """
    Create dataframe with cleaned CMSA data
    Method to be improved
    """
    # Drop duplicates
    cmsa_df_small = cmsa_df.drop_duplicates()
    
    # Fill missing values with zero
    cmsa_df_small = cmsa_df_small.fillna(0)
    
    # Long to wide format
    cmsa_df_wide = cmsa_df_small.pivot_table(index = ["datetime"], columns = "location_id", values = "total_count").reset_index()
    
    # Fill (some) newly created NaNs with previous value
    df_processed = cmsa_df_wide.set_index('datetime').resample('15min').ffill(limit = 1)  ## to discuss
    
    #df_processed = df_processed.fillna(0)  ## Temporary, later use interpolation instead?
    #df_processed = df_processed.interpolate(method = 'linear', axis = 0, limit = 16) 
        
    return df_processed


def fill_cmsa_gaps_csv(df, cmsa_vp, cmsa_ac, cmsa_ws, start_learnset): 
    
    # Pre-process csv data
    cmsa_vp = cmsa_vp.rename(columns = {'Geselecteerde week': 'GAVM-02-Stadhouderskade'})
    cmsa_ac = cmsa_ac.rename(columns = {'Geselecteerde week': 'GAAM-01-AlbertCuypstraat'})
    cmsa_ws = cmsa_ws.rename(columns = {'GACM-02 Nieuwendijk t.h.v.218': 'GACM-02'})

    cmsa_vpac = pd.merge(cmsa_vp, cmsa_ac, on = 'Time')
    cmsa_vpac['datetime'] = pd.to_datetime(cmsa_vpac['Time']).dt.tz_localize("Europe/Amsterdam", ambiguous = "NaT")
    cmsa_vpac = cmsa_vpac.drop('Time', axis = 1)
    cmsa_vpac = cmsa_vpac[~cmsa_vpac.datetime.duplicated(keep = False)] # drop DST data

    cmsa_ws['datetime'] = pd.to_datetime(cmsa_ws['Time'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize("Europe/Amsterdam", ambiguous = "infer")

    cmsa_ws['CMSA-GAKH-01'] = cmsa_ws['CMSA-GAKH-01 Kalverstraat t.h.v. 1']
    cmsa_ws['CMSA-GAKH-01'][cmsa_ws['datetime'] < '2021-04-10 00:00:00'] = cmsa_ws['GKS-01-Kalverstraat Kalverstraat t.h.v. 1'][cmsa_ws['datetime'] < '2021-04-10 00:00:00']
    cmsa_ws = cmsa_ws[['GACM-02', 'CMSA-GAKH-01', 'datetime']]

    cmsa_backup = pd.merge(cmsa_vpac, cmsa_ws, on = 'datetime', how = 'outer') #.set_index('datetime')
    cmsa_backup = cmsa_backup[cmsa_backup['datetime'] >= start_learnset] 
    
    # Select which datapoint to use: we prefer database over csv, except when database is NaN
    cmsa_backup['origin'] = 0
    df['origin'] = 1
    df['origin'][df.isna().any(axis=1)] = -1

    df_fill = pd.concat([df.reset_index(), cmsa_backup])
    df_fill = df_fill.sort_values(by=['datetime', 'origin'])
    df_fill = df_fill.drop_duplicates('datetime', keep = 'last').set_index('datetime')
    
    print(df_fill['origin'].value_counts())
    df_fill = df_fill.drop(['origin'], axis = 1)
    
    return df_fill


def get_future_df(start_pred, predict_days):
    
    """
    Get empty dataframe for days in the future
    """
    datetime_predict = pd.date_range(start_pred, periods = 24*4*predict_days, freq='15min')
    df = pd.DataFrame(data = {'datetime' : datetime_predict}).set_index('datetime')
    
    return df


def add_time_variables(df):
    """
    Create dummy variables from weekday and hour and add these to the dataframe for modelling. 
    """
    df = df.reset_index()
    
    # add weekday and hour dummies
    df['weekday'] = pd.Categorical(df['datetime'].dt.weekday)
    df['hour'] =  pd.Categorical(df['datetime'].dt.hour)
    
    weekday_dummies = pd.get_dummies(df[['weekday']], prefix='weekday_')
    hour_dummies = pd.get_dummies(df[['hour']], prefix='hour_')
    
    df_time = df.merge(weekday_dummies, left_index = True, right_index = True)
    df_time = df_time.merge(hour_dummies, left_index = True, right_index = True)
    
    # add weekend features
    df_time['weekend'] = 0 
    df_time['weekend'][df_time['weekday'].isin([5,6])] = 1
    
    # add cyclical time features
    df_time['minutes'] = df_time['datetime'].dt.hour * 60 + df_time['datetime'].dt.minute
    df_time['sin_time'] = np.sin(2 * np.pi * df_time['minutes'] / (24 * 60)) 
    df_time['cos_time'] = np.cos(2 * np.pi * df_time['minutes'] / (24 * 60)) 
        
    df_time = df_time.set_index('datetime')
    
    return df_time


def get_variables_df(df, covid_df, holiday_df, vacation_df, df_knmi, x_cols, lag_df, lag_df2):
    """
    Get dataframes with all input variables, by merging dataframes per source
    For some sources, data needs to be resampled to 15 minutes first. 
    """
    # add time vars
    df_time = add_time_variables(df)
    
    # add weather vars
    df_knmi = df_time.merge(df_knmi, on = 'datetime', how = "left")
    
    # add covid var
    covid_df_q = covid_df.resample('15min').ffill().reset_index() #('1H').ffill()
    df_covid = df_knmi.merge(covid_df_q, on = 'datetime', how = "left")
    
    # add holidays
    holiday_df_q = holiday_df.resample('15min').ffill().reset_index()
    df_holiday = df_covid.merge(holiday_df_q, on = 'datetime', how = "left")   
    
    # add vacations
    vacation_df_q = vacation_df.resample('15min').ffill().reset_index()
    df_final = df_holiday.merge(vacation_df_q, on = 'datetime', how = "left")   
    
    # set index
    df_final = df_final.set_index('datetime')
       
    # define X
    df_X = df_final[x_cols]
    
    # fill missing values # to do: interpolation in preprocessing steps
    #df_X = df_X.fillna(0)
    
    # add lagged features
    df_X = pd.merge(df_X, lag_df, left_index = True, right_index = True, how = 'left')
    df_X = pd.merge(df_X, lag_df2, left_index = True, right_index = True, how = 'left')
    x_cols_final = x_cols + list(lag_df) + list(lag_df2)
    
    return df_X, x_cols_final


### FUNCTIONS - modelling (linear regression)

def train_model_lm(df_X_train, df_y_train):     
    
    """
    Train linear regression model
    """
    # remove rows with NaN in X and y
    df_y_train = df_y_train[~df_X_train.isna().any(axis = 1)] 
    df_X_train = df_X_train[~df_X_train.isna().any(axis = 1)]

    # fit model
    model = linear_model.LinearRegression()
    model.fit(df_X_train, df_y_train)
    
    return model


def test_model_lm(model, df_X_test):
    """
    Run trained linear regression model on the prediction or test set
    """
    # remove rows with na
    df_X_test = df_X_test[~df_X_test.isna().any(axis = 1)]
    
    # basic prediction
    prediction = model.predict(df_X_test)
    
    # prediction adjustments (positive integers)
    prediction[prediction < 0] = 0
    prediction = prediction.astype(int)
       
    return prediction


def train_predict_model_lm(df_y_train, df_X_train, df_y_predict, df_X_predict, location):
    """
    Train & predict linear regression model per location
    """
    # get one location
    df_y_train_loc = df_y_train[[location]]
    
    # fit model
    model = train_model_lm(df_X_train, df_y_train_loc)

    # predict
    df_y_predict_temp = df_y_predict[~df_X_predict.isna().any(axis = 1)]
    df_y_predict_temp['predict_lm_' + location] = test_model_lm(model, df_X_predict)
    df_y_predict = pd.merge(df_y_predict, df_y_predict_temp[['predict_lm_' + location]], 
                            left_index = True, right_index = True, how = 'left')
    
    return df_y_predict


### FUNCTIONS - modelling (xgboost)

def train_model_xg(df_X_train, df_y_train):     
    
    """
    Train XGBoost model
    """
    # fit model
    model =  xgb.XGBRegressor(objective ='reg:squarederror', 
                              learning_rate = 0.1,  
                              n_estimators = 50, 
                              max_depth = 6,
                              subsample = 0.8,
                              colsample_bytree = 0.75, 
                             )
                              
    model.fit(df_X_train, df_y_train)
    
    return model


def test_model_xg(model, df_X_test):
    """
    Run trained XGBoost model on the prediction or test set
    (till now same as test_model_lm)
    """
    # basic prediction
    prediction = model.predict(df_X_test)
    
    # prediction adjustments (positive integers)
    prediction[prediction < 0] = 0
    prediction = prediction.astype(int)
       
    return prediction


def train_predict_model_xg(df_y_train, df_X_train, df_y_predict, df_X_predict, location):
    """
    Train & predict xgboost model per location
    """
    # get one location
    df_y_train_loc = df_y_train[[location]]
    
    # drop columns not for this location (df_X_train en df_X_predict)
    cols = [c for c in df_X_train.columns if c.lower()[:3] != 'lag'] + list(['lag_' + location, 'lag2_' + location])
    df_X_train = df_X_train[cols]
    df_X_predict = df_X_predict[cols]
    
    # fit model
    model = train_model_xg(df_X_train, df_y_train_loc)
    #xgb.plot_importance(model)

    # predict
    df_y_predict['predict_xg_' + location] = test_model_xg(model, df_X_predict)
       
    return df_y_predict


### FUNCTIONS - post operational modelling

def prepare_final_dataframe(df):
    """
    Preparing for the output to be stored in the database. 
    We make the dataframe hourly, and and an integer. 
    In the process, column names are also corrected. 
    """
    ## resample to get hourly prediction: sum 4 quarters of each hour
    #df_h = df.resample('H').sum()
    df_h = df
    
    # get datetime back as column
    df_h = df_h.reset_index()
    
    # back to long format
    df_long = pd.melt(df_h, id_vars = "datetime", var_name = "prediction_id", value_name = "total_count_predict")
    
    return df_long


def create_api_output(df, df_raw, start_pred, predict_days):
    """
    Creating the final output for in the database, for the API. 
    We add the current datetime as model creation time.
    In addition, we are adding area type and location name back from the initial dataframe.
    Finally, we are selecting the amount of days. 
    """
    # prepare dataframes for merging   
    df_raw = df_raw.rename(columns={'area': 'area_type'})  # keep? 
    sensor_info = df_raw[["location_id", "area_type", "location_name"]].tail(50000).drop_duplicates()
    df['location_id'] = df['prediction_id'].str.lstrip('predict_xg_')  # ('predict_lm_')
    
    # merge dataframes, to add area info
    df_api = df.merge(sensor_info, how = "left")
    
    # add current datetime
    df_api['created_at'] = datetime.now(tz = pytz.timezone('Europe/Amsterdam'))
    
    # select only the prediction period
    end_pred = start_pred + timedelta(days = predict_days)
    df_api_pred = df_api[(df_api['datetime'] >= start_pred) & (df_api['datetime'] < end_pred)] 
           
    return df_api_pred


def add_versions_to_store(df, model_version, data_version):
    """
    Creating the final output for in the database, for complete storage of model outputs. 
    We add the model and data version. 
    """
    df['model_version'] = model_version
    df['data_version'] = data_version
    
    return df


### FUNCTIONS - backtesting 

def create_train_and_test_set(df, start_week, start_date):
    """
    Split the dataset in training and testset by giving the date of the week. 
    Optional, also provide a start date
    """
    
    df = df.reset_index()
    
    split_idx = df[df['datetime'] == start_week].index[0]
    start_idx = df[df['datetime'] == start_date].index[0]  
    assert split_idx > start_idx, "Starting date occurs later, than the test week"
    week_length_in_minutes = 24 * 7 * 60
    num_blocks = int(week_length_in_minutes / 15)
    train_set = df[start_idx:split_idx].reset_index(drop=True)
    test_set = df[split_idx:split_idx + num_blocks].reset_index(drop=True)
    
    train_set = train_set.set_index('datetime')
    test_set = test_set.set_index('datetime')
    
    return train_set, test_set


def train_predict_model_lag(df, my_col, my_shift):
    """
    Baseline model for comparison purposes. 
    The prediction for coming week is exactly last week. 
    This is done by created a lagged feature of 7 days. 
    """
    new_col = 'predict_lag_' + my_col
    df[new_col] = df[my_col].shift(my_shift)
    return df


def find_start_of_weeks(df):
    """
    This functions return all the start of weeks in the dataset
    """
    df = df.reset_index()
    mon_start = df['datetime'][(df['datetime'].dt.weekday == 0) & (df['datetime'].dt.hour == 0) & (df['datetime'].dt.minute == 0)] 
    return list(mon_start)


def evaluate(pred, ground_truth, print_metrics=True):
    """
    Evaluate a prediction and ground_truth.
    """
    # fill NaNs with zeroes
    pred = pred.fillna(0)
    ground_truth = ground_truth.fillna(0)
    
    # Calculate root mean squared error
    rmse = mean_squared_error(ground_truth, pred, squared=False)
      
    # Calculate root mean squared error only for crowded moments (p80)   
    busy = np.percentile(ground_truth, 80)
    ground_truth_busy = ground_truth[ground_truth > busy]
    pred_busy = pred[ground_truth > busy]
    rmse_busy = mean_squared_error(ground_truth_busy, pred_busy, squared=False)
    
    if print_metrics:
        print(f"Root mean squared error: {rmse.round(1)}")
        print(f"Root mean squared error (crowded): {rmse_busy.round(1)}")
    return rmse, rmse_busy 


def evaluate_n_last_weeks_lag(num_weeks, df_y, lag_df, starting_datetime, my_locations):
    """
    Take predictions from basic lag model, for n times. Finally, the mean RMSE over the n weeks is provided
    """
    df_y = df_y.reset_index()
    stats = []
    preds = pd.DataFrame()
    weeks = sorted(find_start_of_weeks(df_y)[:-1], reverse=True) # Delete the last one because we might miss some data there
    for idx in range(num_weeks):
        current_week = weeks[idx]
        train_set, test_set = create_train_and_test_set(df_y, current_week, starting_datetime)
        test_set = pd.merge(test_set, lag_df, left_index = True, right_index = True, how = 'left')
        for location in my_locations:   
            stats.append(evaluate(test_set['lag_' + location], test_set[location], print_metrics=False))
        preds = preds.append(test_set)
    rmse, rmse_busy = zip(*stats)
    print(f"Statistics for lag 'model' over {num_weeks} last weeks")
    print(f"(Mean/std) Root mean squared error: {np.mean(rmse).round(1)}/{np.std(rmse).round(1)}")
    print(f"(Mean/std) Root mean squared error (crowded): {np.mean(rmse_busy).round(1)}/{np.std(rmse_busy).round(1)}")
    
    return preds

    
def train_evaluate_n_last_weeks_lm(num_weeks, df_y, df_X, starting_datetime, my_locations):
    """
    Train linear regresion model on the last n fully available week in the dataset.
    The model trains on a traininset from the starting date until the "test" week and
    does this for n times. Finally, the mean RMSE over the n weeks is provided
    """
    df_y = df_y.reset_index()
    df_X = df_X.reset_index()
    stats = []
    preds = pd.DataFrame()
    weeks = sorted(find_start_of_weeks(df_y)[:-1], reverse=True) ## delete the last one because we might miss some data there
    for idx in range(num_weeks):
        current_week = weeks[idx]
        df_y_train_bt, df_y_test = create_train_and_test_set(df_y, current_week, starting_datetime)
        df_X_train_bt, df_X_test = create_train_and_test_set(df_X, current_week, starting_datetime)
        for location in my_locations:
            df_y_test = train_predict_model_lm(df_y_train_bt, df_X_train_bt, df_y_test, df_X_test, location)
            stats.append(evaluate(df_y_test['predict_lm_' + location], df_y_test[location], print_metrics=False))
        preds = preds.append(df_y_test)
    rmse, rmse_busy = zip(*stats)
    #print(rmse)
    #print(rmse_busy)
    print(f"Statistics for linear regression model over {num_weeks} last weeks")
    print(f"(Mean/std) Root mean squared error: {np.mean(rmse).round(1)}/{np.std(rmse).round(1)}")
    print(f"(Mean/std) Root mean squared error (crowded): {np.mean(rmse_busy).round(1)}/{np.std(rmse_busy).round(1)}")
    
    return preds.sort_index()
  

def train_evaluate_n_last_weeks_xg(num_weeks, df_y, df_X, starting_datetime, my_locations):
    """
    Train XGBoost model on the last n fully available week in the dataset.
    The model trains on a traininset from the starting date until the "test" week and
    does this for n times. Finally, the mean RMSE over the n weeks is provided
    """
    df_y = df_y.reset_index()
    df_X = df_X.reset_index()
    stats = []
    preds = pd.DataFrame()
    weeks = sorted(find_start_of_weeks(df_y)[:-1], reverse=True) ## delete the last one because we might miss some data there
    for idx in range(num_weeks):
        current_week = weeks[idx]
        df_y_train_bt, df_y_test = create_train_and_test_set(df_y, current_week, starting_datetime)
        df_X_train_bt, df_X_test = create_train_and_test_set(df_X, current_week, starting_datetime)
        for location in my_locations:
            df_y_test = train_predict_model_xg(df_y_train_bt, df_X_train_bt, df_y_test, df_X_test, location)
            stats.append(evaluate(df_y_test['predict_xg_' + location], df_y_test[location], print_metrics=False))
        preds = preds.append(df_y_test)
    rmse, rmse_busy = zip(*stats)
    print(f"Statistics for XGBoost model over {num_weeks} last weeks")
    print(f"(Mean/std) Root mean squared error: {np.mean(rmse).round(1)}/{np.std(rmse).round(1)}")
    print(f"(Mean/std) Root mean squared error (crowded): {np.mean(rmse_busy).round(1)}/{np.std(rmse_busy).round(1)}")
    
    return preds.sort_index()
