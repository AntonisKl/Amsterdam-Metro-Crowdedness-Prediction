# Resono 1 week predictions

# ### Preparations

# Main libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import copy
import prediction_model_helpers as h

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
import matplotlib.dates as mdates

# Plotting
import matplotlib
import matplotlib.pyplot as plt


def prepare_data(env, resono_data_dir, file_name, freq, predict_period, start_prediction, n_samples_day, Y_names, 
                 target, start_learnset):
    '''
    Prepare data for Resono predictions 
    '''
    
    # ### 1. Get data
    
    # Database connection
    engine_azure, table_names = h.prepare_engine(env)
    
    # Minio connection
    spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
    # Which locations to include (based on druktebeeld inclusion)
    if Y_names == "all":
        Y_inc = select_locations(engine_azure)

    # Read in historic Resono data (daily visitor counts)
    os.chdir(resono_data_dir) 
    resono_df_raw = pd.read_csv(file_name)
    if isinstance(Y_names, list):
        resono_df_raw = resono_df_raw[resono_df_raw['Location'].isin(Y_names)]
    
    # Select moment to end operational forecast based on last value of variable to predict in database
    #start_prediction = pd.date_range(resono_df_raw["Date"].max(), periods = 2, freq = freq)[1]
    end_prediction = pd.date_range(start_prediction, periods = predict_period, freq = freq)
    end_prediction = end_prediction[len(end_prediction)-1]

    # COVID data raw
    covid_url = 'https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/date-range/'+ str(start_learnset) + "/" + str(start_prediction)
    covid_df_raw = pd.DataFrame(requests.get(url = covid_url).json()['data'])

    # Holidays data raw
    holidays_data_raw = Netherlands().holidays(2020) + Netherlands().holidays(2021) 
    
    # Vacation data raw
    # define vacations (we might want to put this in a database table in the future)
    kerst_19 = pd.DataFrame(data = {'date': pd.date_range(date(2019, 12, 21), periods = 7*2 + 2, freq='1d')})
    voorjaar_20 = pd.DataFrame(data = {'date': pd.date_range(date(2020, 2, 15), periods = 9, freq='1d')})
    mei_20 = pd.DataFrame(data = {'date': pd.date_range(date(2020, 4, 25), periods = 9, freq='1d')})
    zomer_20 = pd.DataFrame(data = {'date': pd.date_range(date(2020, 7, 4), periods = 7*6 + 2, freq='1d')})
    herfst_20 = pd.DataFrame(data = {'date': pd.date_range(date(2020, 10, 10), periods = 9, freq='1d')})
    kerst_20 = pd.DataFrame(data = {'date': pd.date_range(date(2020, 12, 19), periods = 7*2 + 2, freq='1d')})
    voorjaar_21 = pd.DataFrame(data = {'date': pd.date_range(date(2021, 2, 20), periods = 9, freq='1d')})
    mei_21 = pd.DataFrame(data = {'date': pd.date_range(date(2021, 5, 1), periods = 9, freq='1d')})
    zomer_21 = pd.DataFrame(data = {'date': pd.date_range(date(2021, 7, 10), periods = 7*6 + 2, freq='1d')})
    herfst_21 = pd.DataFrame(data = {'date': pd.date_range(date(2021, 10, 16), periods = 9, freq='1d')})
    kerst_21 = pd.DataFrame(data = {'date': pd.date_range(date(2021, 12, 25), periods = 7*2 + 2, freq='1d')})
    
    vacation_df_raw = kerst_19.append([voorjaar_20, mei_20, zomer_20, herfst_20, kerst_20,
                                   voorjaar_21, mei_21, zomer_21, herfst_21, kerst_21])
    
    # KNMI data raw
    # /* is data for whole month (up until available)
    # Example: 06/07 whole history was saved (date refers to moment of storage, not data itself)
    knmi_obs_minio = spark.read.format("json").load("s3a://knmi-knmi/topics/knmi-observations/2021/06/*/*")
    knmi_obs_df_raw = knmi_obs_minio.toPandas()
    knmi_pred_minio = spark.read.format("json").option("header", "true").load("s3a://knmi-knmi/topics/knmi/2021/06/*/*.json.gz",
                                                                              sep = ";")
    knmi_pred_df_raw = knmi_pred_minio.limit(5000000).toPandas()
    
    # ### 2. Preprocess data

    # Resono data preprocessed
    resono_df = h.preprocess_resono_hourly_counts_data(resono_df_raw, freq, end_prediction)

    if Y_names == "all":
    # Select locations that are included in druktebeeld
        resono_df = resono_df[resono_df.columns[resono_df.columns.isin(Y_inc)]]
    
    # COVID data preprocessed
    covid_df = h.preprocess_covid_data(covid_df_raw, freq, end_prediction)

    # Holiday data preprocessed
    holiday_df = h.preprocess_holidays_data(holidays_data_raw, freq, end_prediction)
    
    # Vacation data preprocessed
    vacation_df = h.preprocess_vacation_data(vacation_df_raw, freq, end_prediction)
    
    # KNMI data preprocessed
    knmi_obs_df = h.preprocess_knmi_data(knmi_obs_df_raw, freq, end_prediction, resono_hourly_counts = True) 
    knmi_pred_df = h.preprocess_metpre_data(knmi_pred_df_raw, freq, end_prediction, resono_hourly_counts = True)
    # Merge observations with predictions
    knmi_obs_df= knmi_obs_df[knmi_obs_df.index < '2021-06-15 00:00:00']
    knmi_pred_df = knmi_pred_df[knmi_pred_df.index >= '2021-06-15 00:00:00']
    knmi_df = pd.concat([knmi_obs_df, knmi_pred_df])
    
    # Join location-independent data into base df
    base_df = covid_df.join(holiday_df).join(vacation_df).join(knmi_df)
    
    # List of all  (included) location names
    Y_names_all = list(resono_df.columns)

    return base_df, resono_df, resono_df_raw, start_prediction, end_prediction, Y_names_all


def get_resono_predictions(df, resono_df_raw, freq, predict_period, n_samples_day, n_samples_week, Y_name, data_source, 
                           target, outlier_removal, start_learnset,
                           current_model_version, current_data_version, start_prediction, end_prediction, min_train_samples):
    '''
    Get 2h predictions for one Resono location.
    '''
    
    # Give index name
    df.index.name = 'datetime'
    
    # Drop beginning of learnset if data is missing at the beginning
    df = df.loc[df[Y_name].first_valid_index():]
    
    # Impute/drop missing data and substitute outliers
    df = h.clean_data(df, target, Y_name, n_samples_day, cols_to_clean = None, outlier_removal = "no", 
                      resono_hourly_counts = True)
        
    # Add time features
    df = h.add_time_variables(df)

    # Create new features from the data
    df = h.add_lag_variables_week_ahead(df, Y_name, target, predict_period, n_samples_day, n_samples_week)
        
    # filter data based on start learnset
    df = df[start_learnset:]
    
    # No predictions if there is no/not enough training data 
    if (df.empty) | (len(df) < min_train_samples):
        print("No predictions made for " + Y_name + ": not enough training data.")
        
        df_y_predict = h.get_future_df(start_prediction, predict_period, freq)
        
        y_scaler = None

    else:
        # scale dataset
        df_unscaled = df.copy()
        df, y_scaler = h.scale_variables(df, Y_name, target, method = "standard")
        
    # ### 2. Create model dataframes

        df_X_train, df_y_train = h.get_train_df(df, Y_name, start_prediction) 

        df_y_predict = h.get_future_df(start_prediction, predict_period, freq)
        
        df_X_predict = df.drop(Y_name, 1)

        # Select features for prediction period
        df_X_predict = df_X_predict[start_prediction:end_prediction]

    # ### 3. Create operational prediction

        # Prediction model
        model = h.train_model_ridge_regression(df_X_train, df_y_train, Y_name, target)

        df_y_predict[Y_name] = h.test_model_ridge_regression(model, df_X_predict)
                                                         
        # unscale prediction
        if target == "count":
            df_y_predict = h.unscale_y(df_y_predict, y_scaler)


    # ### 4. Prepare output

    final_df = h.prepare_final_dataframe(df_y_predict, resono_df_raw, data_source, target, current_model_version,
                                         current_data_version, resono_hourly_counts = True)

    return df, final_df, y_scaler


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


def get_location_df(base_df, resono_df, Y_name):
    '''
    Get dataframe with target and predictor variables for one location.
    '''
    
    df = pd.concat([resono_df[Y_name], base_df], 1)
    
    return df


def get_location_report_df(final_df, prepared_df, y_scaler, Y_name):
    '''
    Get the dataframe with past observations and predictions for one location.
    '''
    
    # Select predictions
    prediction = final_df[final_df["Location"] == Y_name].set_index('datetime')['total_count_predicted']
    
    # Select average of the past 4 weeks (part of previous week missing)
    average_past_weeks = h.unscale_y(prepared_df, y_scaler)[Y_name + '_prev_weeks_avg']
    
    # Name the series
    prediction.name = "Voorspelling"
    average_past_weeks.name = "Gemiddelde afgelopen weken"
    
    # Combine past observations and predictions as separate columns
    report_df = pd.concat([average_past_weeks, prediction], 1)
    
    # Filter on prediction period
    report_df = report_df[report_df.index.isin(prediction.index)]
    
    return report_df

        
def get_report_plot_hourly(report_df, legend, Y_name, report_dir, week_label):
    '''
    Get a graph of the predictions compared to past observations on an hourly rate.
    '''
    
    # Parameters
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams.update({'axes.titlesize': 18,
                     'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'axes.labelpad': 8.0
                    })

    # Date format
    xfmt = mdates.DateFormatter('%a \n%b %d')

    # Plot past + prediction
    fig, ax = plt.subplots()
    p1 = ax.plot(report_df.iloc[:, 1], color = 'black')
    p2 = ax.plot(report_df.iloc[:, 0], color = 'grey', ls = '--')

    # Plot peak moment
    peak_idx = np.where(report_df['Voorspelling'] == report_df['Voorspelling'].max())[0][0]
    peak_time = str(report_df.iloc[peak_idx, :].name)
    p3 = ax.plot(report_df.iloc[peak_idx, :].name, report_df.iloc[peak_idx, 1], 'ro', ms = 10)

    # Plot legend
    if legend == "yes":
        ax.legend([p1, p2, p3], loc='upper right', bbox_to_anchor=(1.4, 1),
                 labels = [report_df.columns[1], report_df.columns[0], 'Verwachte piek: ' + peak_time])
    
    # Plot title
    ax.set_title(str("Voorspelde drukte " + Y_name))
    
    # Y-axis title
    plt.ylabel("Aantal bezoekers per uur")

    # Set date format
    ax.xaxis.set_major_formatter(xfmt)

    # Remove plot part of outline
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Set figure size
    fig.set_size_inches(10, 6)
    
    # Store figure
    os.makedirs(report_dir + week_label, exist_ok=True) 
    os.chdir(report_dir + week_label)
    fig_name = Y_name.replace(" ", "_") + "_resono_week_ahead_prediction_hourly.png"
    plt.savefig(fig_name, bbox_inches='tight', dpi = 500)
    
    
def get_report_plot_daily(report_df, Y_name, report_dir, week_label):
    '''
    Get a graph of the predictions compared to past observations on a daily rate with week and weekend splitted.
    '''
    
    # Get daily sum and split between weekdays and weekend
    daily = report_df.resample('D').sum()
    daily_weekend = daily[daily.index.weekday.isin([5, 6])]
    daily_weekdays = daily[daily.index.weekday.isin([0, 1, 2, 3, 4])]
    
    # Get the mean for weekdays/weekend
    daily_weekdays_avg = daily_weekdays.mean(axis = 0).to_frame().reset_index()
    daily_weekend_avg = daily_weekend.mean(axis = 0).to_frame().reset_index()
    
    # Rename columns
    daily_weekdays_avg.columns = ['Week', 'Count']
    daily_weekend_avg.columns = ['Week', 'Count']
        
    # Convert average past 4 weeks to week numbers and prediction to current week number
    daily_weekdays_avg['Week'] = [str(int(week_label)-3) + "-" + str(int(week_label)-1), week_label]
    daily_weekend_avg['Week'] = [str(int(week_label)-3) + "-" + str(int(week_label)-1), week_label]
    
    # Parameters
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams.update({'axes.titlesize': 14,
                     'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'axes.labelpad': 8.0
                    })
    
    # Initialize figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8,4), dpi = 100, frameon = False, sharey = True, 
                                     constrained_layout = True)

    # Title
    fig.suptitle(Y_name, fontsize = 18)
    
    # Weekdays figure
    ax1.bar(x = daily_weekdays_avg['Week'], height = daily_weekdays_avg['Count'], color = ['green', 'lightgreen'])
    ax1.set_title('Doordeweeks (ma-vr)')
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Aantal Resono counts per dag")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.grid(axis = 'y', color = 'lightgrey')

    # Weekend figure
    ax2.bar(x =  daily_weekend_avg['Week'], height = daily_weekend_avg['Count'], color = ['green', 'lightgreen'])
    ax2.set_title('Weekend (za-zo)')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.grid(axis = 'y', color = 'lightgrey')
    ax2.set_xlabel("Week")

    # Store figure
    os.makedirs(report_dir + week_label, exist_ok=True) 
    os.chdir(report_dir + week_label)
    fig_name = Y_name.replace(" ", "_") + "_resono_week_ahead_prediction_daily.png"
    plt.savefig(fig_name, bbox_inches='tight', dpi = 500)
    

    
    
    
    
    
    
    
