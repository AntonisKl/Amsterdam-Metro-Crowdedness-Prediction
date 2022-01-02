import pandas as pd
import geopandas as gpd
import numpy as np
import os
from sqlalchemy import create_engine
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from shapely import wkt
from datetime import datetime, timedelta, date
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import requests

import matplotlib.pyplot as plt
#import contextily as cx --> gives error?

def get_prediction_df():
    """
    Return the prediction dataframe (date- and hours only)
    """
    this_year = date.today().isocalendar()[0]
    this_week = date.today().isocalendar()[1]
    firstdayofweek = datetime.strptime(f'{this_year}-W{int(this_week )}-1', "%Y-W%W-%w").date()
    prediction_date_range = pd.date_range(first_date, periods=8, freq='D')
    prediction_date_range_hour = pd.date_range(prediction_date_range.min(), prediction_date_range.max(), freq='h').delete(-1)
    return prediction_date_range_hour

def get_vacations():
    """
    Retrieves vacations in the Netherlands from the Government of the Netherlands (Rijksoverheid) and returns
    the list of dates that are vacation dates
    """
    
    vacations_url = 'https://opendata.rijksoverheid.nl/v1/sources/rijksoverheid/infotypes/schoolholidays?output=json'
    vacations_raw = requests.get(url = vacations_url).json()
    
    df_vacations = pd.DataFrame(columns={'vacation', 'region', 'startdate', 'enddate'})

    for x in range(0, len(vacations_raw)): # Iterate through all vacation years
        for y in range(0, len(vacations_raw[0]['content'][0]['vacations'])): # number of vacations in a year
            dates = pd.DataFrame(vacations_raw[x]['content'][0]['vacations'][y]['regions'])
            dates['vacation'] = vacations_raw[x]['content'][0]['vacations'][y]['type'].strip() # vacation name
            dates['school_year'] = vacations_raw[x]['content'][0]['schoolyear'].strip() # school year
            df_vacations = df_vacations.append(dates)

    filtered = df_vacations[(df_vacations['region']=='noord') | (df_vacations['region']=='heel Nederland')]
    
    vacations_date_only = pd.DataFrame(columns={'date'})
    
    for x in range(0, len(filtered)):
        df_temporary = pd.DataFrame(data = {'date':pd.date_range(filtered.iloc[x]['startdate'], filtered.iloc[x]['enddate'], freq='D') + pd.Timedelta(days=1)})
        vacations_date_only = vacations_date_only.append(df_temporary)
    
    vacations_date_only['date'] = vacations_date_only['date'].apply(lambda x: x.date)
    vacations_date_only['date'] = vacations_date_only['date'].astype('datetime64[ns]')
    
    # Since the data from Rijksoverheid starts from school year 2019-2020, add the rest of 2019 vacations manually!
    kerst_18 = pd.DataFrame(data = {'date': pd.date_range(date(2019, 1, 1), periods = 6, freq='1d')})
    voorjaar_19 = pd.DataFrame(data = {'date': pd.date_range(date(2019, 2, 16), periods = 9, freq='1d')})
    mei_19 = pd.DataFrame(data = {'date': pd.date_range(date(2019, 4, 27), periods = 9, freq='1d')})
    zomer_19 = pd.DataFrame(data = {'date': pd.date_range(date(2019, 7, 13), periods = 7*6 + 2, freq='1d')})
    
    vacations_date_only = vacations_date_only.append([kerst_18, voorjaar_19, mei_19, zomer_19])
    
    return vacations_date_only

def get_events():
    """
    Event data from static file. We can store events in the database in the near future. When possible, we can get it from an API.
    """
    
    events = pd.read_excel('events_zuidoost.xlsx', sheet_name='Resultaat', header=1)
    
    # Clean
    events.dropna(how='all', inplace=True)
    events.drop(events.loc[events['Datum']=='Niet bijzonder evenementen zijn hierboven niet meegenomen.'].index, inplace=True)
    events.drop(events.loc[events['Locatie'].isna()].index, inplace=True)
    events.drop(events.loc[events['Locatie']=='Overig'].index, inplace=True)
    events['Datum'] = events['Datum'].astype('datetime64[ns]')
    
    # Fix location names
    events['Locatie'] = events['Locatie'].apply(lambda x: x.strip()) # Remove spaces
    events['Locatie'] = np.where(events['Locatie'] == 'Ziggo dome', 'Ziggo Dome', events['Locatie'])
    events['Locatie'] = np.where(events['Locatie'] == 'Ziggo Dome (2x)', 'Ziggo Dome', events['Locatie'])
    
    # Get events from 2019 from static file
    events = events[events['Datum'].dt.year>=2019].copy()
    events.reset_index(inplace=True)
    events.drop(columns=['index'], inplace=True)
    events
    
    # Add 2020-present events manually
    events = events.append({'Datum':datetime(2020, 1, 19)}, ignore_index=True) # Ajax - Sparta
    events = events.append({'Datum':datetime(2020, 2, 2)}, ignore_index=True) # Ajax - PSV
    events = events.append({'Datum':datetime(2020, 2, 16)}, ignore_index=True) # Ajax - RKC
    events = events.append({'Datum':datetime(2020, 1, 3)}, ignore_index=True) # Ajax - AZ
    
    # Euro 2021
    events = events.append({'Datum':datetime(2021, 6, 13)}, ignore_index=True) # EURO 2020 Nederland- Oekraïne
    events = events.append({'Datum':datetime(2021, 6, 17)}, ignore_index=True) # EURO 2020 Nederland- Oostenrijk
    events = events.append({'Datum':datetime(2021, 6, 21)}, ignore_index=True) # EURO 2020 Noord-Macedonië - Nederland
    events = events.append({'Datum':datetime(2021, 6, 26)}, ignore_index=True) # EURO 2020 Wales - Denemarken
    
    return events

def merge_csv_json(bestemming_csv, herkomst_csv, bestemming_json, herkomst_json):
    bestemming = pd.concat([bestemming_csv, bestemming_json]).copy()
    herkomst = pd.concat([herkomst_csv, herkomst_json]).copy() 

    return [bestemming, herkomst]

def merge_bestemming_herkomst(bestemming, herkomst):
    bestemming.rename(columns={'AantalReizen':'Uitchecks', 
                                                  'UurgroepOmschrijving (van aankomst)':'UurgroepOmschrijving',
                                                  'AankomstHalteCode':'HalteCode',
                                                  'AankomstHalteNaam':'HalteNaam'}, inplace=True)
    herkomst.rename(columns={'AantalReizen':'Inchecks', 
                                                  'UurgroepOmschrijving (van vertrek)':'UurgroepOmschrijving',
                                                  'VertrekHalteCode':'HalteCode',
                                                  'VertrekHalteNaam':'HalteNaam'}, inplace=True)
    
    merged = pd.merge(left=bestemming, right=herkomst, 
                   left_on=['Datum', 'UurgroepOmschrijving', 'HalteNaam'], 
                   right_on=['Datum', 'UurgroepOmschrijving', 'HalteNaam'],
                   how='outer')  
    
    return merged

def preprocess_gvb_data_for_modelling(gvb_df, station):
    df = gvb_df[gvb_df['HalteNaam']==station].copy()
    
    # create datetime column
    df['datetime'] = df['Datum'].astype('datetime64[ns]')
    df['hour'] = df['UurgroepOmschrijving'].apply(lambda x: int(x[:2]))
       
    # add time indications
    df['week'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
        
    # drop duplicates and sort
    df_ok = df.drop_duplicates()
    
    # sort values and reset index
    df_ok = df_ok.sort_values(by = 'datetime')
    df_ok = df_ok.reset_index(drop = True)
    
    # drop unnecessary columns
    df_ok.drop(columns=['Datum', 'UurgroepOmschrijving', 'HalteNaam'], inplace=True)
    
    # rename columns
    df_ok.rename(columns={'Inchecks':'check-ins', 'Uitchecks':'check-outs'}, inplace=True)
                         
    return df_ok

def preprocess_knmi_data_hour(df_raw):
    """
    Prepare the raw knmi data for modelling.
    We rename columns and resample from 60min to 15min data.
    Also, we will create a proper timestamp.
    Documentation: https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
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
    df["datetime"] = pd.to_datetime(df['date'], format='%Y%m%dT%H:%M:%S.%f') + pd.to_timedelta(df["hour"] - 1, unit = 'hours')
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam")
    df = df.sort_values(by = "datetime", ascending = True)
    df = df.reset_index(drop = True)
    df['date'] = df['datetime'].dt.date
    df['date'] = df['date'].astype('datetime64[ns]')
    df['hour'] -= 1
    
    # drop unwanted columns
    df = df.drop(['datetime', 'weather_code', 'station_code'], axis = 'columns')
    
    # divide some columns by ten (because using 0.1 degrees C etc. as units)
    col10 = ["wind_speed", "wind_gust", "temperature", "temperature_min", "dew_point_temperature",
             "radiation_duration", "precipitation_duration", "precipitation_h", "pressure"]
    df[col10] = df[col10] / 10

    return df

def preprocess_metpre_data(df_raw):
    """
    To be filled
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
    df["datetime"] = df["datetime"] + pd.to_timedelta(1, unit = 'hours')  ## klopt dan beter, maar waarom?
    df = df.sort_values(by = "datetime", ascending = True)
    df = df.reset_index(drop = True)
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam")
    # new column: forecast created on
    df["offset_h"] = df["offset"].astype(float)
    #df["datetime_predicted"] = df["datetime"] - pd.to_timedelta(df["offset_h"], unit = 'hours')
    # select only data after starting datetime
    #df = df[df['datetime'] >= start_ds]  # @me: move this to query later
    # select latest prediction # logisch voor prediction set, niet zozeer voor training set
    df = df.sort_values(by = ['datetime', 'offset_h'])
    df = df.drop_duplicates(subset = 'datetime', keep = 'first')
    # drop unwanted columns
    df = df.drop(['tijd', 'tijd_nl', 'loc',
                  'icoon', 'samenv', 'ico',
                  'cape', 'cond',  'luchtdmmhg', 'luchtdinhg',
                  'windkmh', 'windknp', 'windrltr', 'wind_force',
                  'gustb', 'gustkt', 'gustkmh', 'wind_gust',  # deze zitten er niet in voor 14 juni
                  'hw', 'mw', 'lw',
                  'offset', 'offset_h',
                  'gr_w'], axis = 'columns', errors = 'ignore')
    # set datatypes of weather data to float
    df = df.set_index('datetime')
    df = df.astype('float64').reset_index()
    # cloud cover similar to observations (0-9) & sight, but not really the same thing
    df['cloud_cover'] =  df['clouds'] / 12.5
    df['sight'] =  df['sight_m'] / 333
    df.drop(['clouds', 'sight_m'], axis = 'columns')
    # go from hourly to quarterly values
    df_hour = df.set_index('datetime').resample('1h').ffill(limit = 11)
    # later misschien smoothen? lijkt nu niet te helpen voor voorspelling
    #df_smooth = df_15.apply(lambda x: savgol_filter(x,17,2))
    #df_smooth = df_smooth.reset_index()
    df_hour = df_hour.reset_index()
    df_hour['date'] = df_hour['datetime'].dt.date
    df_hour['date'] = df_hour['date'].astype('datetime64[ns]')
    df_hour['hour'] = df_hour['datetime'].dt.hour
    
    return df_hour  # df_smooth

def preprocess_covid_data(df_raw):
    # Put data to dataframe
    df_raw_unpack = df_raw.T['NLD'].dropna()
    df = pd.DataFrame.from_records(df_raw_unpack)    # Add datetime column
    df['datetime'] = pd.to_datetime(df['date_value'])    # Select columns
    df_sel = df[['datetime', 'stringency']]    # extend dataframe to 14 days in future (based on latest value)
    dates_future = pd.date_range(df['datetime'].iloc[-1], periods = 14, freq='1d')
    df_future = pd.DataFrame(data = {'datetime': dates_future,
                                       'stringency': df['stringency'].iloc[-1]})    # Add together and set index
    df_final = df_sel.append(df_future.iloc[1:])
    df_final = df_final.set_index('datetime')
    return df_final


def preprocess_holiday_data(holidays):    
    df = pd.DataFrame(holidays, columns=['Date', 'Holiday'])
    df['Date'] = df['Date'].astype('datetime64[ns]')
    return df

def interpolate_missing_values(data_to_interpolate):
    df = data_to_interpolate.copy()
    random_state_value = 1 # Ensure reproducability
        
    # Train check-ins interpolator
    checkins_interpolator_cols = ['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-outs']
    checkins_interpolator_targets = ['check-ins']
    
    X_train = df.dropna()[checkins_interpolator_cols]
    y_train = df.dropna()[checkins_interpolator_targets]

    checkins_interpolator = RandomForestRegressor(random_state=random_state_value)
    checkins_interpolator.fit(X_train, y_train)
    
    # Train check-outs interpolator
    checkouts_interpolator_cols = ['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-ins']
    checkouts_interpolator_targets = ['check-outs']
    
    X_train = df.dropna()[checkouts_interpolator_cols]
    y_train = df.dropna()[checkouts_interpolator_targets]
    
    checkouts_interpolator = RandomForestRegressor(random_state=random_state_value)
    checkouts_interpolator.fit(X_train, y_train)
    
    # Select rows which need interpolation
    df_to_interpolate = df.drop(df.loc[(df['check-ins'].isna()==True) & (df['check-outs'].isna()==True)].index)
    
    # Interpolate check-ins
    checkins_missing = df_to_interpolate[(df_to_interpolate['check-outs'].isna()==False) & (df_to_interpolate['check-ins'].isna()==True)].copy()
    checkins_missing['check-ins'] = checkins_interpolator.predict(checkins_missing[['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-outs']])
    
    # Interpolate check-outs
    checkouts_missing = df_to_interpolate[(df_to_interpolate['check-ins'].isna()==False) & (df_to_interpolate['check-outs'].isna()==True)].copy()
    checkouts_missing['check-outs'] = checkouts_interpolator.predict(checkouts_missing[['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-ins']])
    
    # Insert interpolated values into main dataframe
    for index, row in checkins_missing.iterrows():
        df.loc[df.index==index, 'check-ins'] = row['check-ins']
        
    for index, row in checkouts_missing.iterrows():
        df.loc[df.index==index, 'check-outs'] = row['check-outs']
        
    return df 

def get_crowd_last_week(df, row):
    week_ago = row['datetime'] - timedelta(weeks=1)
    subset_with_hour = df[(df['datetime']==week_ago) & (df['hour']==row['hour'])]
    
    # If crowd from last week is not available at exact date- and hour combination, then get average crowd of last week.
    subset_week_ago = df[(df['year']==row['year']) & (df['week']==row['week']) & (df['hour']==row['hour'])]
    
    checkins_week_ago = 0
    checkouts_week_ago = 0
    
    if len(subset_with_hour) > 0: # return crowd from week ago at the same day/time (hour)
        checkins_week_ago = subset_with_hour['check-ins'].mean()
        checkouts_week_ago = subset_with_hour['check-outs'].mean() 
    elif len(subset_week_ago) > 0: # return average crowd the hour group a week ago
        checkins_week_ago = subset_week_ago['check-ins'].mean()
        checkouts_week_ago = subset_week_ago['check-outs'].mean()
        
    return [checkins_week_ago, checkouts_week_ago]

def get_train_test_split(df):
    """
    Create train and test split for 1-week ahead models. This means that the last week of the data will be used
    as a test set and the rest will be the training set.
    """
    
    most_recent_date = df['datetime'].max()
    last_week = pd.date_range(df.datetime.max()-pd.Timedelta(7, unit='D')+pd.DateOffset(1), df['datetime'].max())
    
    train = df[df['datetime']<last_week.min()]
    test = df[(df['datetime']>=last_week.min()) & (df['datetime']<=last_week.max())]
    
    return [train, test]

def get_train_val_test_split(df):
    """
    Create train, validation, and test split for 1-week ahead models. This means that the last week of the data will be used
    as a test set, the second-last will be the validation set, and the rest will be the training set.
    """
    
    most_recent_date = df['datetime'].max()
    last_week = pd.date_range(df.datetime.max()-pd.Timedelta(7, unit='D')+pd.DateOffset(1), df['datetime'].max())
    two_weeks_before = pd.date_range(last_week.min()-pd.Timedelta(7, unit='D'), last_week.min()-pd.DateOffset(1))
    
    train = df[df['datetime']<two_weeks_before.min()]
    validation = df[(df['datetime']>=two_weeks_before.min()) & (df['datetime']<=two_weeks_before.max())]
    test = df[(df['datetime']>=last_week.min()) & (df['datetime']<=last_week.max())]
    
    return [train, validation, test]

def get_future_df(features, gvb_data, covid_stringency, holidays, vacations, weather, events):
    """
    Create empty data frame for predictions of the target variable for the specfied prediction period
    """
    
    this_year = date.today().isocalendar()[0]
    this_week = date.today().isocalendar()[1]
    firstdayofweek = datetime.strptime(f'{this_year}-W{int(this_week )}-1', "%Y-W%W-%w").date()
    prediction_date_range = pd.date_range(firstdayofweek, periods=8, freq='D')
    prediction_date_range_hour = pd.date_range(prediction_date_range.min(), prediction_date_range.max(), freq='h').delete(-1)
    
    # Create variables
    df = pd.DataFrame({'datetime':prediction_date_range_hour})
    df['hour'] = df.apply(lambda x: x['datetime'].hour, axis=1)
    df['week'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    df['stringency'] = covid_stringency
    df['datetime'] = df.apply(lambda x: x['datetime'].date(), axis=1)
    df['datetime'] = df['datetime'].astype('datetime64[ns]')
    
    # Set holidays, vacations, and events
    df['holiday'] = np.where((df['datetime'].isin(holidays['Date'].values)), 1, 0)
    df['vacation'] = np.where((df['datetime'].isin(vacations['date'].values)), 1, 0)
    
    # Get events from database in future!
    df['planned_event'] = np.where((df['datetime'].isin(events['Datum'].values)), 1, 0)
    
    # Set forecast for temperature, rain, and wind speed.
    df = pd.merge(left=df, right=weather.drop(columns=['datetime']), left_on=['datetime', 'hour'], right_on=['date', 'hour'], how='left')
    df.drop(columns=['date'], inplace=True)
    
    # Set recent crowd
    df[['check-ins_week_ago', 'check-outs_week_ago']] = df.apply(lambda x: get_crowd_last_week(gvb_data, x), axis=1, result_type="expand")
    
    if not 'datetime' in features:
        features.append('datetime') # Add datetime to make storing in database easier

    return df[features]

def train_random_forest_regressor(X_train, y_train, X_val, y_val, hyperparameters=None):
    if hyperparameters == None:
        model = RandomForestRegressor(random_state=1).fit(X_train, y_train)
    else:
        model = RandomForestRegressor(**hyperparameters, random_state=1).fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    r_squared =  metrics.r2_score(y_val, y_pred)
    mae = metrics.mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    return [model, r_squared, mae, rmse]

def merge_gvb_with_datasources(gvb, weather, covid, holidays, vacations, events):
    gvb_merged = pd.merge(left=gvb, right=weather, left_on=['datetime', 'hour'], right_on=['date', 'hour'], how='left')
    gvb_merged.drop(columns=['date'], inplace=True)
    
    gvb_merged = pd.merge(gvb_merged, covid['stringency'], left_on='datetime', right_index=True, how='left')
    
    gvb_merged['holiday'] = np.where((gvb_merged['datetime'].isin(holiday_df['Date'].values)), 1, 0)
    gvb_merged['vacation'] = np.where((gvb_merged['datetime'].isin(vacations['date'].values)), 1, 0)
    gvb_merged['planned_event'] = np.where((gvb_merged['datetime'].isin(events['Datum'].values)), 1, 0)
    
    return gvb_merged

def predict(model, X_predict):
    y_predict = model.predict(X_predict.drop(columns=['datetime']))
    predictions = X_predict.copy()
    predictions['check-ins_predicted'] = y_predict[:,0]
    predictions['check-outs_predicted'] = y_predict[:,1]
    return predictions

def set_station_type(df, static_gvb):
    stationtypes = static_gvb[['arrival_stop_code', 'type']]
    return pd.merge(left=df, right=stationtypes, left_on='HalteCode', right_on='arrival_stop_code', how='inner')

def merge_bestemming_herkomst_stop_level(bestemming, herkomst):
    bestemming.rename(columns={'AantalReizen':'Uitchecks', 
                                                  'UurgroepOmschrijving (van aankomst)':'UurgroepOmschrijving',
                                                  'AankomstHalteCode':'HalteCode',
                                                  'AankomstHalteNaam':'HalteNaam'}, inplace=True)
    herkomst.rename(columns={'AantalReizen':'Inchecks', 
                                                  'UurgroepOmschrijving (van vertrek)':'UurgroepOmschrijving',
                                                  'VertrekHalteCode':'HalteCode',
                                                  'VertrekHalteNaam':'HalteNaam'}, inplace=True)
    
    merged = pd.merge(left=bestemming, right=herkomst, 
                   left_on=['Datum', 'UurgroepOmschrijving', 'HalteCode', 'HalteNaam'], 
                   right_on=['Datum', 'UurgroepOmschrijving', 'HalteCode', 'HalteNaam'],
                   how='outer')  
    
    return merged

def get_crowd_last_week_stop_level(df, row):
    week_ago = row['datetime'] - timedelta(weeks=1)
    subset_with_hour = df[(df['type_metro']==row['type_metro']) & (df['type_tram/bus']==row['type_tram/bus']) &
                          (df['datetime']==week_ago) & (df['hour']==row['hour'])]
    
    # If crowd from last week is not available at exact date- and hour combination, then get average crowd of last week.
    subset_week_ago = df[(df['type_metro']==row['type_metro']) & (df['type_tram/bus']==row['type_tram/bus']) &
                         (df['year']==row['year']) & (df['week']==row['week']) & (df['hour']==row['hour'])]
    
    checkins_week_ago = 0
    checkouts_week_ago = 0
    
    if len(subset_with_hour) > 0: # return crowd from week ago at the same day/time (hour)
        checkins_week_ago = subset_with_hour['check-ins'].mean()
        checkouts_week_ago = subset_with_hour['check-outs'].mean() 
    elif len(subset_week_ago) > 0: # return average crowd the hour group a week ago
        checkins_week_ago = subset_week_ago['check-ins'].mean()
        checkouts_week_ago = subset_week_ago['check-outs'].mean()
        
    return [checkins_week_ago, checkouts_week_ago]

"""
Below are old functions which are not used for the prediction models.
"""    

def preprocess_gvb_data(df):
    
    # create datetime column
    df['date'] = pd.to_datetime(df['Datum'])
    df['time'] = df['UurgroepOmschrijving (van aankomst)'].astype(str).str[:5]
    df['datetime'] = df['date'].astype(str) + " " + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'])
       
    # add time indications
    df['week'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    
    # fix amount of travels
    df['AantalReizen'] = pd.to_numeric(df['AantalReizen']) 
    df = df.loc[df['AantalReizen'].notna()]
    
    # lon and lat numeric
    df['AankomstLat'] = pd.to_numeric(df['AankomstLat'])
    df['AankomstLon'] = pd.to_numeric(df['AankomstLon'])
    
    # drop duplicates and sort
    df_ok = df.drop_duplicates()
    
    # sort values and reset index
    df_ok = df_ok.sort_values(by = 'datetime')
    df_ok = df_ok.reset_index(drop = True)
    
    # drop 
    df_ok = df_ok.drop(['Datum', 'UurgroepOmschrijving (van aankomst)'], axis = 1)
    
    # rename columns
    df_ok = df_ok.rename(columns = {'AankomstHalteCode': 'arrival_stop_code',
                                   'AankomstHalteNaam': 'arrival_stop_name',
                                   'AankomstLat': 'arrival_lat',
                                   'AankomstLon': 'arrival_lon',
                                   'AantalReizen': 'count'}) 
                         
    return df_ok


def clean_names_and_types(df_ok):
    
    # Clean types
    df_ok['arrival_stop_code_num'] = pd.to_numeric(df_ok['arrival_stop_code'], errors = 'coerce')
    df_ok['type'] = 'tram/bus'
    df_ok['type'][df_ok['arrival_stop_code_num'].isna()] = 'metro'
    df_ok['type'][df_ok['arrival_stop_name'].isin(['[[ Onbekend ]]', 'Overig'])] = 'onbekend'
    
    # Clean stop names
    df_ok['arrival_stop_name_fix'] = df_ok['arrival_stop_name'] 
    df_ok['arrival_stop_name_fix'] = df_ok['arrival_stop_name_fix'].replace(['Atat?rk','Belgi?plein', 'Burg. R?ellstraat', "Haarl'meerstation", 'Anth. Moddermanstraat', 'Fred. Hendrikplants.', 'G. v.Ledenberchstraat', 
                                                                         'Gaasperplas (uitstap)', 'Jan v.Galenstraat','Lumi?restraat', 'Nw.Willemsstraat', 'P.Oosterhuisstraat', 'Pr. Irenestraat', 'VU medisch centrum'],
                                                                        ['Atatürk','Belgiëplein', 'Burg. Röellstraat','Haarlemmermeerstation',  'Ant. Moddermanstraat', 'Frederik Hendrikplantsoen', 'G.v. Ledenberchstraat',
                                                                         'Gaasperplas', 'Jan van Galenstraat', 'Lumierestraat',   'Nw. Willemsstraat', 'P. Oosterhuisstraat', 'Prinses Irenestraat', 'VU Medisch Centrum'])    
    
    # Create stop group name
    df_ok['arrival_stop_groupname'] = df_ok['arrival_stop_name_fix'] + " (" + df_ok['type'] + ")"  
    
    return df_ok


def create_static_gvb_table(df_ok):
    
    # Set start point
    df_ok_new = df_ok[df_ok['datetime'] >= '2020-10-01 00:00:00']
    
    # Get unique locations
    df_ok_static = df_ok_new[['arrival_stop_code', 'arrival_stop_name',
                          'arrival_lat', 'arrival_lon', 
                          'type', 'arrival_stop_groupname']]
    df_ok_static = df_ok_static.drop_duplicates(['arrival_stop_code']).reset_index(drop=True) 
    
    # Remove rows which are not a station
    df_ok_static = df_ok_static[~df_ok_static['arrival_stop_code'].isin(['[[ Onb', None])]
    
    # Include druktebeeld columns
    df_ok_static['include_druktebeeld'] = 1
    df_ok_static['crowd_threshold_low'] = np.nan
    df_ok_static['crowd_threshold_high'] = np.nan
    
    return df_ok_static


def add_geo_static_gvb_table(df_ok_static):
    
    # Get point geometry
    gdf_ok_static = gpd.GeoDataFrame(df_ok_static, 
                                 geometry = gpd.points_from_xy(df_ok_static.arrival_lat, df_ok_static.arrival_lon))
    
    # Include group geometry
    gdf_ok_static["n_codes"] = 1
    df_ok_static_group = gdf_ok_static.groupby(["arrival_stop_groupname"]).agg({'arrival_lat': 'mean', 
                                                                                'arrival_lon': 'mean',
                                                                                'n_codes': 'sum'}).reset_index()
    gdf_ok_static_group = gpd.GeoDataFrame(df_ok_static_group, 
                                 geometry = gpd.points_from_xy(df_ok_static_group.arrival_lat, df_ok_static_group.arrival_lon))
    gdf_ok_static2 = pd.merge(gdf_ok_static, gdf_ok_static_group, 
                              on = 'arrival_stop_groupname', suffixes = ("", "_group"), how = 'left')
    
    # Remove help columns, and change order of columns  
    gdf_static = gdf_ok_static2[['arrival_stop_code', 'arrival_stop_name', 'type', 'geometry', 
                                 'arrival_stop_groupname', 'n_codes_group', 'geometry_group',
                                 'include_druktebeeld', 'crowd_threshold_low', 'crowd_threshold_high']]
    
    return gdf_static


def add_max_values(df_ok, gdf_static):
    
    # Max 2019
    df_2019 = df_ok[(df_ok['year'] == 2019)] 
    print(len(df_2019["arrival_stop_code"].unique()))
    df_2019_wide = df_2019.pivot_table(index = ['datetime'], columns = "arrival_stop_code", values = "count")
    df_2019_max = pd.DataFrame(df_2019_wide.max(axis=0), columns = ['max_count_2019'] ).reset_index()
    
    # Max Q4 2020
    df_ok_new = df_ok[df_ok['datetime'] >= '2020-10-01 00:00:00']
    print(len(df_ok_new["arrival_stop_code"].unique()))
    df_new_wide = df_ok_new.pivot_table(index = ['datetime'], columns = "arrival_stop_code", values = "count")
    df_new_max = pd.DataFrame(df_new_wide.max(axis=0), columns = ['max_count_2020_Q4'] ).reset_index()
    
    # Add to static table
    df_max = pd.merge(df_2019_max, df_new_max, how = 'outer')
    gdf_static_thresh1 = pd.merge(gdf_static, df_max, how = 'left', on = 'arrival_stop_code')
    print(len(gdf_static_thresh1["arrival_stop_code"].unique()))
    
    return gdf_static_thresh1


def add_lines_per_stop(df_ref, gdf_static_thresh1):
    
    # Add lines to static table
    df_ref['PuntCode_float'] = df_ref['PuntCode'].astype(float)
    gdf_static_thresh1['arrival_stop_code_num'] = pd.to_numeric(gdf_static_thresh1['arrival_stop_code'], errors = 'coerce')
    df_mapped = pd.merge(gdf_static_thresh1, df_ref, left_on = 'arrival_stop_code_num', right_on = 'PuntCode_float', how = 'left')
    
    # Map lines to type: bus, tram or metro
    df_mapped['count_bus'] = 1
    df_mapped['count_bus'][df_mapped['type'] == 'metro'] = 0
    df_mapped['count_bus'][df_mapped['LijnCode'] < 30] = 0
    df_mapped['count_bus'][df_mapped['LijnCode'].isin([15, 18, 21, 22])] = 1

    df_mapped['count_tram'] = 1
    df_mapped['count_tram'][df_mapped['type'] == 'metro'] = 0
    df_mapped['count_tram'][df_mapped['count_bus'] == 1] = 0

    df_mapped['count_metro'] = 0
    df_mapped['count_metro'][df_mapped['type'] == 'metro'] = 1
    
    # Aggregate counts
    df_mapped_group = df_mapped[['arrival_stop_code', 'count_bus', 'count_tram', 'count_metro']].groupby('arrival_stop_code').sum().reset_index()
    df_mapped_group2 = df_mapped.groupby('arrival_stop_code')['LijnCode'].apply(list).reset_index()
    df_mapped_group2.rename(columns = {'LijnCode': 'lines'}, inplace = True)
    
    # Add to static table
    df_mapped_group_total = pd.merge(df_mapped_group, df_mapped_group2, how = 'left', on = 'arrival_stop_code')
    gdf_static_thresh2 = pd.merge(gdf_static_thresh1, df_mapped_group_total, how = 'left', on = 'arrival_stop_code')
    
    # Remove column
    del gdf_static_thresh2['arrival_stop_code_num']
    
    return gdf_static_thresh2
    
    
def prepare_accessibility_data(haltes_raw):
    
    haltes = haltes_raw[haltes_raw['quaystatus'] == 'available']
    #haltes = haltes[haltes['transportmode'] != 'ferry'].reset_index(drop=True)  

    #haltes['quaycode_short'] = pd.to_numeric(haltes['quaycode'].str[-4:])
    haltes['quaycode_short'] = haltes['quaycode'].str[-5:]
    haltes['quayname'][haltes['quayname'] == 'Jan v.Galenstraat'] = 'Jan van Galenstraat' # staat er anders in voor metro dan tram/bus
    
    # Remove unwanted locations
    #print(haltes[haltes.duplicated(['quaycode_short'], keep = False)].sort_values(by='quaycode_short'))
    haltes = haltes[haltes.quaycode != 'NL:Q:57002171'] # Zeilstraat, dubbel
    haltes = haltes[haltes.quaycode != 'NL:Q:57005070'] # CS, dubbel
    haltes = haltes[haltes.quaycode != 'NL:Q:57005080'] # CS, dubbel
    haltes = haltes[haltes.quaycode != 'NL:Q:57005090'] # CS, dubbel
    haltes = haltes[haltes.quaycode_short != '20010'] # bestaat niet meer
    haltes = haltes[haltes.quaycode_short != '20020'] # bestaat niet meer
    #print(haltes[haltes.duplicated(['quaycode_short'], keep = False)].sort_values(by='quaycode_short'))
    
    return haltes


def create_trambus_df(haltes):
    
    # Select rows and columns
    haltes_tb = haltes[haltes['transportmode'].isin(['tram', 'bus'])]
    haltes_tb = haltes_tb[['quaycode_short', 'quayname', 
                 'boardingpositionwidth', 'alightingpositionwidth', 'narrowestpassagewidth',
                 'baylength', 'liftedpartlength', 'kerbheight', 'lift']]
    haltes_tb = haltes_tb.rename({'quaycode_short': 'arrival_stop_code'}, axis = 1)
        
    # Group
    haltes_tb_group = haltes_tb.groupby('arrival_stop_code').sum().reset_index()
    
    # Add lift column
    haltes_tb['lift'][haltes_tb['arrival_stop_code'].isin(['05070', '05080', '05090'])] = False
    haltes_tb_group = haltes_tb_group.merge(haltes_tb[['arrival_stop_code', 'lift']].drop_duplicates(),
                                        on = 'arrival_stop_code', how = 'left')
    
    return haltes_tb_group


def create_metro_df(haltes, gdf_static_thresh2):
    
    # Select rows
    haltes_m = haltes[haltes['transportmode'] == 'metro'].reset_index(drop=True)  
    
    # Group
    haltes_m['arrival_stop_groupname'] = haltes_m['quayname'] + " (" + haltes_m['transportmode'] + ")"   
    haltes_m_group = haltes_m.groupby('arrival_stop_groupname').sum().reset_index()
    
    # Add lift
    haltes_m_lift = haltes_m[['arrival_stop_groupname', 'lift']].drop_duplicates()
    haltes_m_group = haltes_m_group.merge(haltes_m_lift, on = 'arrival_stop_groupname', how = 'left' )
    
    # Add arrival_stop_code
    haltes_m_group = pd.merge(haltes_m_group, gdf_static_thresh2[['arrival_stop_code',  'arrival_stop_groupname']], 
                             how = 'left', on = 'arrival_stop_groupname', 
                            copy = False)
    
    # Select columns
    haltes_m_group = haltes_m_group[['arrival_stop_code',  
                 'boardingpositionwidth', 'alightingpositionwidth', 'narrowestpassagewidth',
                 'baylength', 'liftedpartlength',  'kerbheight', 'lift']]
    
    return haltes_m_group


def add_accessibility(haltes_m_group, haltes_tb_group, gdf_static_thresh2):
    
    # Merge tram/bus with metro
    haltes_group = pd.concat([haltes_tb_group, haltes_m_group]).drop_duplicates()
    
    # Add to static df
    gdf_static_thresh3 = pd.merge(gdf_static_thresh2, haltes_group,
                             how = 'left', on = 'arrival_stop_code')
    
    return gdf_static_thresh3

def get_vacations_manually_specified():
    """
    This is an old version of getting vacations by manually specifying them.
    """    
    
    kerst_18 = pd.DataFrame(data = {'Date': pd.date_range(date(2019, 1, 1), periods = 6, freq='1d')})
    voorjaar_19 = pd.DataFrame(data = {'Date': pd.date_range(date(2019, 2, 16), periods = 9, freq='1d')})
    mei_19 = pd.DataFrame(data = {'Date': pd.date_range(date(2019, 4, 27), periods = 9, freq='1d')})
    zomer_19 = pd.DataFrame(data = {'Date': pd.date_range(date(2019, 7, 13), periods = 7*6 + 2, freq='1d')})
    herfst_19 = pd.DataFrame(data = {'Date': pd.date_range(date(2019, 10, 19), periods = 9, freq='1d')})
    kerst_19 = pd.DataFrame(data = {'Date': pd.date_range(date(2019, 12, 21), periods = 7*2 + 2, freq='1d')})
    
    voorjaar_20 = pd.DataFrame(data = {'Date': pd.date_range(date(2020, 2, 15), periods = 9, freq='1d')})
    mei_20 = pd.DataFrame(data = {'Date': pd.date_range(date(2020, 4, 25), periods = 9, freq='1d')})
    zomer_20 = pd.DataFrame(data = {'Date': pd.date_range(date(2020, 7, 4), periods = 7*6 + 2, freq='1d')})
    herfst_20 = pd.DataFrame(data = {'Date': pd.date_range(date(2020, 10, 10), periods = 9, freq='1d')})
    kerst_20 = pd.DataFrame(data = {'Date': pd.date_range(date(2020, 12, 19), periods = 7*2 + 2, freq='1d')})
    
    voorjaar_21 = pd.DataFrame(data = {'Date': pd.date_range(date(2021, 2, 20), periods = 9, freq='1d')})
    mei_21 = pd.DataFrame(data = {'Date': pd.date_range(date(2021, 5, 1), periods = 9, freq='1d')})
    zomer_21 = pd.DataFrame(data = {'Date': pd.date_range(date(2021, 7, 10), periods = 7*6 + 2, freq='1d')})
    herfst_21 = pd.DataFrame(data = {'Date': pd.date_range(date(2021, 10, 16), periods = 9, freq='1d')})
    kerst_21 = pd.DataFrame(data = {'Date': pd.date_range(date(2021, 12, 25), periods = 7*2 + 2, freq='1d')})
    
    vacations = pd.concat([kerst_18,
                       voorjaar_19, mei_19, zomer_19, herfst_19, kerst_19,
                       voorjaar_20, mei_20, zomer_20, herfst_20, kerst_20,
                       voorjaar_21, mei_21, zomer_21, herfst_21, kerst_21])
    
    return vacations
