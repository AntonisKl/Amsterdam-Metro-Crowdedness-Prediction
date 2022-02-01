import builtins
import configparser
import gzip
import json
import math
import os
import re
from datetime import datetime, timedelta, date
from glob import glob

import geopy.distance
import numpy as np
import pandas as pd
import requests
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

config = configparser.ConfigParser()
config.optionxform = str
config.read('config.ini')
config_use_normalized_visitors = config['DEFAULT'].getboolean('UseNormalizedVisitors')
config_use_event_station_distance = config['DEFAULT'].getboolean('UseEventStationDistance')
config_include_instagram_events = config['DEFAULT'].getboolean('IncludeInstagramEvents')
config_include_ticketmaster_events = config['DEFAULT'].getboolean('IncludeTicketmasterEvents')
config_use_time_of_events = config['DEFAULT'].getboolean('UseTimeOfEvents')
config_max_hours_before_event = config['DEFAULT'].getint('MaxHoursBeforeEvent')
config_max_minutes_before_event = config['DEFAULT'].getint('MaxMinutesBeforeEvent')
config_max_hours_after_event = config['DEFAULT'].getint('MaxHoursAfterEvent')
config_max_minutes_after_event = config['DEFAULT'].getint('MaxMinutesAfterEvent')
config_use_covid_stringency = config['DEFAULT'].getboolean('UseCOVIDStringency')
config_use_covid_measures = config['DEFAULT'].getboolean('UseCOVIDMeasures')
config_use_covid_cases = config['DEFAULT'].getboolean('UseCOVIDCases')
config_use_covid_deaths = config['DEFAULT'].getboolean('UseCOVIDDeaths')


def get_gvb_data(file_prefix):
    gvb_df = None

    for filepath in glob('gvb/**/**/**/{}*.csv'.format(file_prefix)):
        if not os.path.isfile(filepath) or not os.path.getsize(filepath) > 0:
            continue

        current_df = pd.read_csv(filepath, sep=';')
        if gvb_df is None:
            gvb_df = current_df
        else:
            gvb_df = gvb_df.append(current_df)

    return gvb_df


def get_gvb_data_json_checkout(gvb_df):
    files = glob('gvb/**/**/**/*.json.gz')
    dfs = []
    for file in files:
        if not os.path.isfile(file) or not os.path.getsize(file) > 0:
            continue
        dfs.append(pd.read_json(file, compression="gzip", lines=True))

    gvb_json_df = pd.concat(dfs)

    return gvb_df.append(gvb_json_df)

def get_gvb_data_json_checkin(gvb_df):
    files = glob('gvb-herkomst/**/**/**/*.json.gz')
    dfs = []
    for file in files:
        if not os.path.isfile(file) or not os.path.getsize(file) > 0:
            continue
        dfs.append(pd.read_json(file, compression="gzip", lines=True))

    gvb_json_df = pd.concat(dfs)

    return gvb_df.append(gvb_json_df)



# Ramon Dop - 12 jan 2021
def get_covid_measures():
    url = "https://www.ecdc.europa.eu/en/publications-data/download-data-response-measures-covid-19"
    response = requests.get(url)
    # if this gives an error it is because the url has changed or the name of the file has changed on the website, please
    # consult https://www.ecdc.europa.eu/en/publications-data/download-data-response-measures-covid-19
    if response.status_code == 200:
        csv_url = re.findall(
            r'https://www\.ecdc\.europa\.eu/sites/default/files/documents/response_graphs_data_\d{4}-\d{2}-\d{2}.csv',
            response.text)[0]
        measures_raw = pd.read_csv(csv_url)
    else:
        measures_raw = pd.read_csv(r'response_graphs_data_2021-12-20.csv')

    measures_raw = measures_raw.fillna(0)
    measures_raw['date_end'] = measures_raw['date_end'].replace(0, datetime.today().strftime('%Y-%m-%d'))
    measures_rawNL = measures_raw[measures_raw['Country'] == 'Netherlands']

    measures_rawNL["date"] = measures_rawNL.apply(
        lambda x: pd.date_range(x["date_start"], x["date_end"]), axis=1
    )
    measures_rawNL = (
        measures_rawNL.explode("date")
            .drop(columns=["date_start", "date_end"])
    )
    measures_rawNL['dummy'] = 1
    df_out = pd.pivot(measures_rawNL, index='date', columns="Response_measure", values="dummy").reset_index()
    df_out.set_index('date', inplace=True)
    covid_df = df_out.where(df_out == 1, other=0)

    return covid_df


# End


def get_covid_cases_deaths():
    df_covid = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv/data.csv')
    df_covid_nl = df_covid[df_covid['geoId'] == 'NL']
    df_covid_nl['datetime'] = pd.to_datetime(df_covid_nl['dateRep']).dt.strftime('%Y-%m-%d')
    covid_cases_deaths_df = df_covid_nl[['datetime', 'cases', 'deaths']]
    # covid_cases_deaths_df['datetime'] = pd.to_datetime(covid_cases_deaths_df['datetime'])
    return covid_cases_deaths_df


def get_knmi_data(path):
    json_obj_list = []
    for filepath in glob(path):
        if not os.path.isfile(filepath) or not os.path.getsize(filepath) > 0:
            continue

        with gzip.open(filepath, 'r') as fin:
            json_obj_list.extend([json.loads(json_obj_str) for json_obj_str in fin])

    return pd.DataFrame.from_records(json_obj_list)


def get_vacations():
    """
    Retrieves vacations in the Netherlands from the Government of the Netherlands (Rijksoverheid) and returns
    the list of dates that are vacation dates
    """

    vacations_url = 'https://opendata.rijksoverheid.nl/v1/sources/rijksoverheid/infotypes/schoolholidays?output=json'
    vacations_raw = requests.get(url=vacations_url).json()

    df_vacations = pd.DataFrame(columns={'vacation', 'region', 'startdate', 'enddate'})

    for x in range(0, len(vacations_raw)):  # Iterate through all vacation years
        for y in range(0, len(vacations_raw[0]['content'][0]['vacations'])):  # number of vacations in a year
            dates = pd.DataFrame(vacations_raw[x]['content'][0]['vacations'][y]['regions'])
            dates['vacation'] = vacations_raw[x]['content'][0]['vacations'][y]['type'].strip()  # vacation name
            dates['school_year'] = vacations_raw[x]['content'][0]['schoolyear'].strip()  # school year
            df_vacations = df_vacations.append(dates)

    filtered = df_vacations[(df_vacations['region'] == 'noord') | (df_vacations['region'] == 'heel Nederland')]

    vacations_date_only = pd.DataFrame(columns={'date'})

    for x in range(0, len(filtered)):
        df_temporary = pd.DataFrame(data={
            'date': pd.date_range(filtered.iloc[x]['startdate'], filtered.iloc[x]['enddate'], freq='D') + pd.Timedelta(
                days=1)})
        vacations_date_only = vacations_date_only.append(df_temporary)

    vacations_date_only['date'] = vacations_date_only['date'].apply(lambda x: x.date)
    vacations_date_only['date'] = vacations_date_only['date'].astype('datetime64[ns]')

    # Since the data from Rijksoverheid starts from school year 2019-2020, add the rest of 2019 vacations manually!
    kerst_18 = pd.DataFrame(data={'date': pd.date_range(date(2019, 1, 1), periods=6, freq='1d')})
    voorjaar_19 = pd.DataFrame(data={'date': pd.date_range(date(2019, 2, 16), periods=9, freq='1d')})
    mei_19 = pd.DataFrame(data={'date': pd.date_range(date(2019, 4, 27), periods=9, freq='1d')})
    zomer_19 = pd.DataFrame(data={'date': pd.date_range(date(2019, 7, 13), periods=7 * 6 + 2, freq='1d')})

    vacations_date_only = vacations_date_only.append([kerst_18, voorjaar_19, mei_19, zomer_19])

    return vacations_date_only


def get_events():
    """
    Event data from static file. We can store events in the database in the near future. When possible, we can get it from an API.
    """

    events = pd.read_excel('events_zuidoost.xlsx', sheet_name='Resultaat', header=1)

    # Clean
    events.dropna(how='all', inplace=True)
    events.drop(events.loc[events['Datum'] == 'Niet bijzonder evenementen zijn hierboven niet meegenomen.'].index,
                inplace=True)
    events.drop(events.loc[events['Locatie'].isna()].index, inplace=True)
    events.drop(events.loc[events['Locatie'] == 'Overig'].index, inplace=True)

    if config_use_time_of_events:
        events.dropna(subset=['Start show'], inplace=True)
        events['Start show'] = events['Start show'].astype(str).apply(lambda time: time.replace('1899-12-30 ', '')[:5])
        events['Einde show'] = events['Einde show'].astype(str).apply(lambda time: time.replace('1899-12-30 ', '')[:5])

    events['Datum'] = events['Datum'].astype('datetime64[ns]')
    if config_use_time_of_events:
        events['Datetime'] = pd.to_datetime(
            events['Datum'].dt.strftime('%Y-%m-%d') + ' ' + events['Start show'].astype(str))
        events['End datetime'] = pd.to_datetime(
            events['Datum'].dt.strftime('%Y-%m-%d') + ' ' + events['Einde show'].astype(str), errors='coerce')

    if config_include_instagram_events:
        # Prepare instagram events
        events_instagram = pd.read_csv('../instagram-event-scraper/events.csv', usecols=[1, 6])
        events_instagram.rename(columns={'location': 'Locatie', 'event_date': 'Datum'}, inplace=True)
        preprocess_events_dates(events_instagram)

        usernames_venues = {'ziggodome': 'Ziggo Dome',
                            'paradisoadam': 'Paradiso',
                            'afaslive': 'Afas Live',
                            'johancruijffarena': 'Arena',
                            'melkwegamsterdam': 'Melkweg',
                            'theatercarre': 'Royal Theater Carré',
                            'beursvanberlageofficial': 'Beurs van Berlage',
                            'concertgebouw': 'Concertgebouw',
                            'olympischstadion': 'Olympisch Stadion'}

        # Convert instagram usernames to venue names
        events_instagram['Locatie'] = events_instagram['Locatie'].apply(lambda username: usernames_venues[username])

        # Combine events datasets
        events = pd.concat([events, events_instagram], axis=0, ignore_index=True)

    if config_include_ticketmaster_events:
        # Prepare ticketmaster events
        for filepath in glob('../ticketmaster-event-fetcher/events*.csv'):
            if not os.path.isfile(filepath) or not os.path.getsize(filepath) > 0:
                continue

            events_ticketmaster = pd.read_csv(filepath, usecols=[1, 2, 3])
            events_ticketmaster.rename(columns={'venue': 'Locatie', 'datetime': 'Datum', 'name': 'Naam evenement'},
                                       inplace=True)
            preprocess_events_dates(events_ticketmaster)

            events_ticketmaster['Locatie'] = np.where(events_ticketmaster['Locatie'] == 'Koninklijk Theater Carre',
                                                      'Royal Theater Carré', events_ticketmaster['Locatie'])

            # Combine events datasets
            events = pd.concat([events, events_ticketmaster], axis=0, ignore_index=True)

    # Fix location names
    events['Locatie'] = events['Locatie'].apply(lambda x: x.strip())  # Remove spaces
    events['Locatie'] = np.where(events['Locatie'] == 'Ziggo dome', 'Ziggo Dome', events['Locatie'])
    events['Locatie'] = np.where(events['Locatie'] == 'Ziggo Dome (2x)', 'Ziggo Dome', events['Locatie'])

    # Get events from 2019 from static file
    events = events[events['Datum'].dt.year >= 2019].copy()
    events.reset_index(inplace=True)
    events.drop(columns=['index'], inplace=True)

    # Add 2020-present events manually
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2020, 1, 19)}, ignore_index=True)  # Ajax - Sparta
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2020, 2, 2)}, ignore_index=True)  # Ajax - PSV
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2020, 2, 16)}, ignore_index=True)  # Ajax - RKC
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2020, 1, 3)}, ignore_index=True)  # Ajax - AZ

    # Euro 2021
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2021, 6, 13)},
                           ignore_index=True)  # EURO 2020 Nederland- Oekraïne
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2021, 6, 17)},
                           ignore_index=True)  # EURO 2020 Nederland- Oostenrijk
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2021, 6, 21)},
                           ignore_index=True)  # EURO 2020 Noord-Macedonië - Nederland
    events = events.append({'Locatie': 'Arena', 'Datum': datetime(2021, 6, 26)},
                           ignore_index=True)  # EURO 2020 Wales - Denemarken

    # Remove duplicate events
    events.drop_duplicates(subset=['Locatie', 'Datum', 'Start show'], inplace=True)

    # Add normalized number of visitors
    # max_num_visitors_per_day = events.groupby(['Datum'])['Aantal bezoekers'].sum().max()
    events['visitors_normalized'] = events['Aantal bezoekers'] / events['Aantal bezoekers'].max()

    if config_use_event_station_distance:
        venues_coords = {
            'Ziggo Dome': (52.313522864427696, 4.9371839278273875),
            'Endemol': (52.31535991263321, 4.934918212484825),
            'AFAS Live': (52.31208301885852, 4.944499883649119),
            'De Toekomst': (52.31302103804334, 4.928368425789856),
            'Arena': (52.31452213292746, 4.94193191229651),
            'Gaasperplas': (52.3929282804607, 4.902166515382845),
            'Paradiso': (52.36241084594603, 4.8839668313372515),
            'Afas Live': (52.31198463100763, 4.9445106122964555),
            'Olympisch Stadion': (52.34344264336677, 4.853012827639222),
            'Melkweg': (52.36483839071882, 4.881184525790839),
            'Royal Theater Carré': (52.36241338668293, 4.90423151229743),
            'Beurs van Berlage': (52.375140777876005, 4.896203485310937),
            'Concertgebouw': (52.35645333735133, 4.8790282122973085),
            'De Waalse Kerk': (52.37110648416711, 4.897230269968681),
            'Lovelee': (52.36487084204627, 4.881842227639652)
        }

        stations_coords = {
            'Centraal Station': (52.379338119590415, 4.9001788162610875),
            'Station Zuid': (52.33988258200082, 4.873346017322016),
            'Station Bijlmer ArenA': (52.311700953723935, 4.947731485309759)
        }

        for station in stations_coords.keys():
            events['Distance (km) from station ' + station] = events['Locatie'].apply(
                lambda venue: geopy.distance.distance(venues_coords[venue], stations_coords[station]).km)

    return events


def preprocess_events_dates(events_df):
    events_df['Datum'] = events_df['Datum'].astype('datetime64[ns]')
    events_df['Datetime'] = events_df['Datum']
    events_df['Start show'] = events_df['Datum'].apply(lambda datetime: datetime.time())
    events_df['Datum'] = events_df['Datum'].apply(
        lambda datetime: datetime.replace(hour=0, minute=0, second=0))


def merge_bestemming_herkomst(bestemming, herkomst):
    bestemming.rename(columns={'AantalReizen': 'Uitchecks',
                               'UurgroepOmschrijving (van aankomst)': 'UurgroepOmschrijving',
                               'AankomstHalteCode': 'HalteCode',
                               'AankomstHalteNaam': 'HalteNaam'}, inplace=True)
    herkomst.rename(columns={'AantalReizen': 'Inchecks',
                             'UurgroepOmschrijving (van vertrek)': 'UurgroepOmschrijving',
                             'VertrekHalteCode': 'HalteCode',
                             'VertrekHalteNaam': 'HalteNaam'}, inplace=True)

    merged = pd.merge(left=bestemming, right=herkomst,
                      left_on=['Datum', 'UurgroepOmschrijving', 'HalteNaam'],
                      right_on=['Datum', 'UurgroepOmschrijving', 'HalteNaam'],
                      how='outer')

    return merged


def preprocess_gvb_data_for_modelling(gvb_df, station):
    df = gvb_df[gvb_df['HalteNaam'] == station].copy()

    # create datetime column
    df['datetime'] = df['Datum'].astype('datetime64[ns]')
    df['hour'] = df['UurgroepOmschrijving'].apply(lambda x: int(x[:2]))

    # add time indications
    df['week'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday

    if config_use_time_of_events:
        df["datetime_full"] = pd.to_datetime(
            df['datetime'].dt.strftime('%Y-%m-%d') + ' '
            + df['UurgroepOmschrijving'].apply(lambda time_interval: time_interval[:5]))

    hours = pd.get_dummies(df['hour'], prefix='hour')
    days = pd.get_dummies(df['weekday'], prefix='weekday')

    df = pd.concat([df, hours, days], axis=1)

    # drop duplicates and sort
    df_ok = df.drop_duplicates()

    # sort values and reset index
    df_ok = df_ok.sort_values(by='datetime')
    df_ok = df_ok.reset_index(drop=True)

    # drop unnecessary columns
    df_ok.drop(columns=['Datum', 'UurgroepOmschrijving'], inplace=True)

    # rename columns
    df_ok.rename(columns={'Inchecks': 'check-ins', 'Uitchecks': 'check-outs'}, inplace=True)

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
    df["datetime"] = pd.to_datetime(df['date'], format='%Y%m%dT%H:%M:%S.%f') + pd.to_timedelta(df["hour"] - 1,
                                                                                               unit='hours')
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam")
    df = df.sort_values(by="datetime", ascending=True)
    df = df.reset_index(drop=True)
    df['date'] = df['datetime'].dt.date
    df['date'] = df['date'].astype('datetime64[ns]')
    df['hour'] -= 1

    # drop unwanted columns
    df = df.drop(['datetime', 'weather_code', 'station_code'], axis='columns')

    df = df.astype(
        {'wind_speed': 'float64', 'wind_gust': 'float64', 'temperature': 'float64', 'temperature_min': 'float64',
         'dew_point_temperature': 'float64', 'radiation_duration': 'float64', 'precipitation_duration': 'float64',
         'precipitation_h': 'float64', 'pressure': 'float64'})

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
                                "temp": "temperature", "windb": "wind_force", "winds": "wind_speed",
                                "gust": "wind_gust", "vis": "sight_m", "neersl": "precipitation_h",
                                "gr": "global_radiation", "tw": "clouds"
                                })
    # drop duplicates
    df = df.drop_duplicates()
    # get proper datetime column
    df["datetime"] = pd.to_datetime(df['tijd'], unit='s', utc=True)
    df["datetime"] = df["datetime"] + pd.to_timedelta(1, unit='hours')  ## klopt dan beter, maar waarom?
    df = df.sort_values(by="datetime", ascending=True)
    df = df.reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam")
    # new column: forecast created on
    df["offset_h"] = df["offset"].astype(float)
    # df["datetime_predicted"] = df["datetime"] - pd.to_timedelta(df["offset_h"], unit = 'hours')
    # select only data after starting datetime
    # df = df[df['datetime'] >= start_ds]  # @me: move this to query later
    # select latest prediction # logisch voor prediction set, niet zozeer voor training set
    df = df.sort_values(by=['datetime', 'offset_h'])
    df = df.drop_duplicates(subset='datetime', keep='first')
    # drop unwanted columns
    df = df.drop(['tijd', 'tijd_nl', 'loc',
                  'icoon', 'samenv', 'ico',
                  'cape', 'cond', 'luchtdmmhg', 'luchtdinhg',
                  'windkmh', 'windknp', 'windrltr', 'wind_force',
                  'gustb', 'gustkt', 'gustkmh', 'wind_gust',  # deze zitten er niet in voor 14 juni
                  'hw', 'mw', 'lw',
                  'offset', 'offset_h',
                  'gr_w'], axis='columns', errors='ignore')
    # set datatypes of weather data to float
    df = df.set_index('datetime')

    type_dict = {}
    for column_name in df.columns:
        type_dict[column_name] = 'float64'
    df = df.astype(type_dict, errors='ignore').reset_index()
    # cloud cover similar to observations (0-9) & sight, but not really the same thing
    df['cloud_cover'] = df['clouds'] / 12.5
    df['sight'] = df['sight_m'] / 333
    df.drop(['clouds', 'sight_m'], axis='columns')
    # go from hourly to quarterly values
    df_hour = df.set_index('datetime').resample('1h').ffill(limit=11)
    # later misschien smoothen? lijkt nu niet te helpen voor voorspelling
    # df_smooth = df_15.apply(lambda x: savgol_filter(x,17,2))
    # df_smooth = df_smooth.reset_index()
    df_hour = df_hour.reset_index()
    df_hour['date'] = df_hour['datetime'].dt.date
    df_hour['date'] = df_hour['date'].astype('datetime64[ns]')
    df_hour['hour'] = df_hour['datetime'].dt.hour

    return df_hour  # df_smooth


def preprocess_covid_data(df_raw):
    # Put data to dataframe
    df_raw_unpack = df_raw.T['NLD'].dropna()
    df = pd.DataFrame.from_records(df_raw_unpack)  # Add datetime column
    df['datetime'] = pd.to_datetime(df['date_value'])  # Select columns
    df_sel = df[['datetime', 'stringency']]  # extend dataframe to 14 days in future (based on latest value)
    dates_future = pd.date_range(df['datetime'].iloc[-1], periods=14, freq='1d')
    df_future = pd.DataFrame(data={'datetime': dates_future,
                                   'stringency': df['stringency'].iloc[-1]})  # Add together and set index
    df_final = df_sel.append(df_future.iloc[1:])
    df_final = df_final.set_index('datetime')
    return df_final


def preprocess_holiday_data(holidays):
    df = pd.DataFrame(holidays, columns=['Date', 'Holiday'])
    df['Date'] = df['Date'].astype('datetime64[ns]')
    return df


def interpolate_missing_values(data_to_interpolate):
    df = data_to_interpolate.copy()
    random_state_value = 1  # Ensure reproducability

    # Train check-ins interpolator

    checkins_interpolator_cols = ['hour', 'year', 'weekday', 'month', 'holiday', 'check-outs']

    if config_use_covid_stringency:
        checkins_interpolator_cols.append('stringency')
    checkins_interpolator_targets = ['check-ins']

    X_train = df.dropna()[checkins_interpolator_cols]
    y_train = df.dropna()[checkins_interpolator_targets]

    checkins_interpolator = RandomForestRegressor(random_state=random_state_value)
    checkins_interpolator.fit(X_train, y_train)

    # Train check-outs interpolator
    checkouts_interpolator_cols = ['hour', 'year', 'weekday', 'month', 'holiday', 'check-ins']
    if config_use_covid_stringency:
        checkouts_interpolator_cols.append('stringency')
    checkouts_interpolator_targets = ['check-outs']

    X_train = df.dropna()[checkouts_interpolator_cols]
    y_train = df.dropna()[checkouts_interpolator_targets]

    checkouts_interpolator = RandomForestRegressor(random_state=random_state_value)
    checkouts_interpolator.fit(X_train, y_train)

    # Select rows which need interpolation
    df_to_interpolate = df.drop(df.loc[(df['check-ins'].isna() == True) & (df['check-outs'].isna() == True)].index)

    # Interpolate check-ins
    checkins_missing = df_to_interpolate[
        (df_to_interpolate['check-outs'].isna() == False) & (df_to_interpolate['check-ins'].isna() == True)].copy()
    checkouts_missing = df_to_interpolate[
        (df_to_interpolate['check-ins'].isna() == False) & (df_to_interpolate['check-outs'].isna() == True)].copy()

    if config_use_covid_stringency:
        checkins_missing['stringency'] = checkins_missing['stringency'].replace(np.nan, 0)
        checkins_missing['check-ins'] = checkins_interpolator.predict(
            checkins_missing[['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-outs']])
        checkouts_missing['stringency'] = checkouts_missing['stringency'].replace(np.nan, 0)
        checkouts_missing['check-outs'] = checkouts_interpolator.predict(
            checkouts_missing[['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-ins']])

    else:
        checkins_missing['check-ins'] = checkins_interpolator.predict(
            checkins_missing[['hour', 'year', 'weekday', 'month', 'holiday', 'check-outs']])
        checkouts_missing['check-outs'] = checkouts_interpolator.predict(
            checkouts_missing[['hour', 'year', 'weekday', 'month', 'holiday', 'check-ins']])

    # Interpolate check-outs
    checkouts_missing = df_to_interpolate[
        (df_to_interpolate['check-ins'].isna() == False) & (df_to_interpolate['check-outs'].isna() == True)].copy()
    checkouts_missing['stringency'] = checkouts_missing['stringency'].replace(np.nan, 0)
    checkouts_missing['check-outs'] = checkouts_interpolator.predict(
        checkouts_missing[['hour', 'year', 'weekday', 'month', 'stringency', 'holiday', 'check-ins']])

    # Insert interpolated values into main dataframe
    for index, row in checkins_missing.iterrows():
        df.loc[df.index == index, 'check-ins'] = row['check-ins']

    for index, row in checkouts_missing.iterrows():
        df.loc[df.index == index, 'check-outs'] = row['check-outs']

    return df


def get_crowd_last_week(df, row):
    week_ago = row['datetime'] - timedelta(weeks=1)
    subset_with_hour = df[(df['datetime'] == week_ago) & (df['hour'] == row['hour'])]

    # If crowd from last week is not available at exact date- and hour combination, then get average crowd of last week.
    subset_week_ago = df[(df['year'] == row['year']) & (df['week'] == row['week']) & (df['hour'] == row['hour'])]

    checkins_week_ago = 0
    checkouts_week_ago = 0

    if len(subset_with_hour) > 0:  # return crowd from week ago at the same day/time (hour)
        checkins_week_ago = subset_with_hour['check-ins'].mean()
        checkouts_week_ago = subset_with_hour['check-outs'].mean()
    elif len(subset_week_ago) > 0:  # return average crowd the hour group a week ago
        checkins_week_ago = subset_week_ago['check-ins'].mean()
        checkouts_week_ago = subset_week_ago['check-outs'].mean()

    return [checkins_week_ago, checkouts_week_ago]


def get_train_test_split(df):
    """
    Create train and test split for 1-week ahead models. This means that the last week of the data will be used
    as a test set and the rest will be the training set.
    """

    most_recent_date = df['datetime'].max()
    last_week = pd.date_range(df.datetime.max() - pd.Timedelta(7, unit='D') + pd.DateOffset(1), df['datetime'].max())

    train = df[df['datetime'] < last_week.min()]
    test = df[(df['datetime'] >= last_week.min()) & (df['datetime'] <= last_week.max())]

    return [train, test]


def get_train_val_test_split(df):
    """
    Create train, validation, and test split for 1-week ahead models. This means that the last week of the data will be used
    as a test set, the second-last will be the validation set, and the rest will be the training set.
    """

    last_week = pd.date_range(df.datetime.max() - pd.Timedelta(7, unit='D') + pd.DateOffset(1), df['datetime'].max())
    two_weeks_before = pd.date_range(last_week.min() - pd.Timedelta(7, unit='D'), last_week.min() - pd.DateOffset(1))

    train = df[df['datetime'] < two_weeks_before.min()]
    validation = df[(df['datetime'] >= two_weeks_before.min()) & (df['datetime'] <= two_weeks_before.max())]
    test = df[(df['datetime'] >= last_week.min()) & (df['datetime'] <= last_week.max())]

    return [train, validation, test]


def get_future_df(features, gvb_data, covid_stringency, measures, covid_cases_deaths, holidays, vacations, weather,
                  events):
    """
    Create empty data frame for predictions of the target variable for the specfied prediction period
    """

    this_year = date.today().isocalendar()[0]
    this_week = date.today().isocalendar()[1]
    firstdayofweek = datetime.strptime(f'{this_year}-W{int(this_week)}-1', "%Y-W%W-%w").date()
    prediction_date_range = pd.date_range(firstdayofweek, periods=8, freq='D')
    prediction_date_range_hour = pd.date_range(prediction_date_range.min(), prediction_date_range.max(),
                                               freq='h').delete(-1)

    # Create variables
    df = pd.DataFrame({'datetime': prediction_date_range_hour})
    df['hour'] = df.apply(lambda x: x['datetime'].hour, axis=1)
    df['week'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    df['stringency'] = covid_stringency
    df['datetime_full'] = df['datetime']
    df['datetime'] = df.apply(lambda x: x['datetime'].date(), axis=1)
    df['datetime'] = df['datetime'].astype('datetime64[ns]')

    # adding sin and cosine features
    df["hour_norm"] = 2 * math.pi * df["hour"] / df["hour"].max()
    df["cos_hour"] = np.cos(df["hour_norm"])
    df["sin_hour"] = np.sin(df["hour_norm"])

    df["month_norm"] = 2 * math.pi * df["month"] / df["month"].max()
    df["cos_month"] = np.cos(df["month_norm"])
    df["sin_month"] = np.sin(df["month_norm"])

    df["weekday_norm"] = 2 * math.pi * df["weekday"] / df["weekday"].max()
    df["cos_weekday"] = np.cos(df["weekday_norm"])
    df["sin_weekday"] = np.sin(df["weekday_norm"])

    # adding dummy variable for peak hour
    df['peak_period'] = 0
    df['peak_period'][df.hour.isin([7, 8, 17, 18])] = 1

    # Set holidays, vacations, and events
    df['holiday'] = np.where((df['datetime'].isin(holidays['Date'].values)), 1, 0)
    df['vacation'] = np.where((df['datetime'].isin(vacations['date'].values)), 1, 0)

    df['planned_event'] = df.apply(lambda row: get_planned_event_value(row, events, gvb_data['HalteNaam'].values[0]),
                                   axis=1)

    # Set forecast for temperature, rain, and wind speed.
    df = pd.merge(left=df, right=weather.drop(columns=['datetime']), left_on=['datetime', 'hour'],
                  right_on=['date', 'hour'], how='left')

    datetime_start_of_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if config_use_covid_measures:
        df = pd.merge(df, measures, how='left', left_on='datetime', right_on='date')
        df[measures.columns] = df[measures.columns].fillna(0)
        df.drop(columns=['date'], inplace=True)

        todays_measures = [df[df['datetime'] == datetime_start_of_today][measures_column].values[0] for measures_column
                           in measures.columns]

        for i, measures_column in enumerate(measures.columns):
            df[measures_column].fillna(
                todays_measures[i], inplace=True)

    # Set recent crowd
    df[['check-ins_week_ago', 'check-outs_week_ago']] = df.apply(lambda x: get_crowd_last_week(gvb_data, x), axis=1,
                                                                 result_type="expand")

    if not 'datetime' in features:
        features.append('datetime')  # Add datetime to make storing in database easier

    if config_use_covid_cases or config_use_covid_deaths:
        df = pd.merge(df, covid_cases_deaths, on='datetime', how='left')

        if config_use_covid_cases:
            df['cases'].fillna(
                df[df['datetime'] == datetime_start_of_today]['cases'].values[0], inplace=True)
        if config_use_covid_deaths:
            df['deaths'].fillna(
                df[df['datetime'] == datetime_start_of_today]['deaths'].values[0], inplace=True)

    return df[features]


def train_random_forest_regressor(X_train, y_train, X_val, y_val, hyperparameters=None):
    if hyperparameters is None:
        model = RandomForestRegressor(random_state=1).fit(X_train, y_train)
    else:
        model = RandomForestRegressor(**hyperparameters, random_state=1).fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r_squared = metrics.r2_score(y_val, y_pred)
    mae = metrics.mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
    return [model, r_squared, mae, rmse]


def get_planned_event_value(gvb_merged_row, events_df, station=None):
    mask = (events_df['Datum'] == gvb_merged_row['datetime'])
    if config_use_time_of_events:
        # each event is affecting the last config_max_hours_before_event hours before it happens
        # and the next config_max_hours_after_event hours after it happens
        before_event_cond = (gvb_merged_row['datetime_full'] >= events_df['Datetime'] - timedelta(
            hours=config_max_hours_before_event, minutes=config_max_minutes_before_event)) & (
                                    gvb_merged_row['datetime_full'] <= events_df['Datetime'])
        after_event_cond = (~events_df['End datetime'].isna()) & (
                gvb_merged_row['datetime_full'] <= events_df['End datetime'] + timedelta(
            hours=config_max_hours_after_event, minutes=config_max_minutes_after_event)) & (
                gvb_merged_row['datetime_full'] >= events_df['End datetime'])
        mask = (before_event_cond) | (after_event_cond)

    event_station_distances = pd.Series(0)
    if config_use_event_station_distance:
        event_station_distances = events_df[mask][
            'Distance (km) from station ' + (station if station else gvb_merged_row['HalteNaam'])]

    if config_use_normalized_visitors:
        visitors_normalized = events_df[mask]['visitors_normalized']
        if len(visitors_normalized) == 0:
            return 0  # No events found

        visitors_normalized = visitors_normalized.dropna()
        if len(visitors_normalized) == 0:
            return 0.5 / (1 + event_station_distances.min())

        event_station_distances = event_station_distances.loc[visitors_normalized.index.values]
        return visitors_normalized.loc[event_station_distances.idxmin()] / (1 + event_station_distances.min())  # number of events * max normalized visitors
    else:
        return 1 / (1 + event_station_distances.min()) if len(events_df[mask]) > 0 else 0


def merge_gvb_with_datasources(gvb, weather, covid, measures, holidays, vacations, events, covid_cases_deaths_df):
    gvb_merged = pd.merge(left=gvb, right=weather, left_on=['datetime', 'hour'], right_on=['date', 'hour'], how='left')
    gvb_merged.drop(columns=['date'], inplace=True)

    if config_use_covid_stringency:
        gvb_merged = pd.merge(gvb_merged, covid['stringency'], left_on='datetime', right_index=True, how='left')
        gvb_merged['stringency'] = gvb_merged['stringency'].fillna(0)

    gvb_merged['holiday'] = np.where((gvb_merged['datetime'].isin(holidays['Date'].values)), 1, 0)
    gvb_merged['vacation'] = np.where((gvb_merged['datetime'].isin(vacations['date'].values)), 1, 0)
    gvb_merged['planned_event'] = gvb_merged.apply(lambda row: get_planned_event_value(row, events), axis=1)

    # check how many NAs at this point
    if config_use_covid_measures:
        # Add COVID measures
        gvb_merged = pd.merge(gvb_merged, measures, how='left', left_on='datetime', right_on='date')
        gvb_merged[measures.columns] = gvb_merged[measures.columns].fillna(0)

    if config_use_covid_cases:
        # Add COVID cases
        covid_cases_deaths_df['cases'] = covid_cases_deaths_df['cases'].fillna(0)
        covid_cases_deaths_df['cases'] = covid_cases_deaths_df['cases'].astype('float64')
        covid_cases_deaths_df['datetime'] = pd.to_datetime(covid_cases_deaths_df['datetime'])
        gvb_merged = pd.merge(gvb_merged, covid_cases_deaths_df.drop(columns=['deaths']), on='datetime', how='left')
        gvb_merged['cases'] = gvb_merged['cases'].fillna(0)

    if config_use_covid_deaths:
        # Add COVID deaths
        covid_cases_deaths_df['deaths'] = covid_cases_deaths_df['deaths'].fillna(0)
        covid_cases_deaths_df['deaths'] = covid_cases_deaths_df['deaths'].astype('float64')
        covid_cases_deaths_df['datetime'] = pd.to_datetime(covid_cases_deaths_df['datetime'])
        gvb_merged = pd.merge(gvb_merged, covid_cases_deaths_df.drop(columns=['cases']), on='datetime', how='left')
        gvb_merged['deaths'] = gvb_merged['deaths'].fillna(0)

    return gvb_merged


def predict(model, X_predict):
    y_predict = model.predict(X_predict.drop(columns=['datetime']))
    predictions = X_predict.copy()
    predictions['check-ins_predicted'] = y_predict[:, 0]
    predictions['check-outs_predicted'] = y_predict[:, 1]
    return predictions


def set_station_type(df, static_gvb):
    stationtypes = static_gvb[['arrival_stop_code', 'type']]
    return pd.merge(left=df, right=stationtypes, left_on='HalteCode', right_on='arrival_stop_code', how='inner')


def merge_bestemming_herkomst_stop_level(bestemming, herkomst):
    bestemming.rename(columns={'AantalReizen': 'Uitchecks',
                               'UurgroepOmschrijving (van aankomst)': 'UurgroepOmschrijving',
                               'AankomstHalteCode': 'HalteCode',
                               'AankomstHalteNaam': 'HalteNaam'}, inplace=True)
    herkomst.rename(columns={'AantalReizen': 'Inchecks',
                             'UurgroepOmschrijving (van vertrek)': 'UurgroepOmschrijving',
                             'VertrekHalteCode': 'HalteCode',
                             'VertrekHalteNaam': 'HalteNaam'}, inplace=True)

    merged = pd.merge(left=bestemming, right=herkomst,
                      left_on=['Datum', 'UurgroepOmschrijving', 'HalteCode', 'HalteNaam'],
                      right_on=['Datum', 'UurgroepOmschrijving', 'HalteCode', 'HalteNaam'],
                      how='outer')

    return merged

def get_busiest_x_hours():
    #todo make it dynamic
    top_promille = 1
    bijlmer_df = gvb_dfs_merged[2].sort_values(by=['datetime_full'])
    for df in gvb_dfs:
        #checkout_bijlmer_df = newbijlmer_df.sort_values(by=['check-outs'])
        checkout_bijlmer_df = df.sort_values(by=['check-outs'])
        checkout_bijlmer_df = checkout_bijlmer_df.drop(columns=['check-ins'])
        checkout_bijlmer_df = checkout_bijlmer_df.dropna(axis='rows')
        last_x = round(len(checkout_bijlmer_df)/1000*top_promille)
        top_x = checkout_bijlmer_df.tail(last_x)
        print(top_x['check-outs'].iloc[0])

        
def find_time_between_peak_and_start_event():  
    # todo make it dynamic
    # FIND AVERAGE TIME BETWEEN PEAK AND EVENT
    # Recommended Average peak is 2 hours
    events_visited = events[events['Aantal bezoekers']>0]
    counter_list = []
    bijlmer_df = gvb_dfs_merged[2].sort_values(by=['datetime_full'])
    newbijlmer_df = bijlmer_df.iloc[:, :-106]

    hours_back = 4
    for date in events_visited['Datetime']:
        idx = newbijlmer_df.index[newbijlmer_df['datetime_full'] == date]
        if (idx > 0) : # if events dont start at a round hour it is ignored for now
            max = 0
            counter = -1 # because at hour starttime of event(minus zero) will always be bigger than 0
            for i in range(hours_back):
                lookup = date - timedelta(hours=i)
                df_comb = newbijlmer_df[newbijlmer_df['datetime_full'] == lookup]
                if (df_comb['check-outs'].values > max):
                    counter = counter + 1
                    max = df_comb['check-outs'].values
            counter_list.append(counter)

    avg_peakhour = sum(counter_list) / len(counter_list)
    print(avg_peakhour)
    
# models: array in format [[<SciKit model object>, <number>, <number>, <number>], ... ]
def log_models(models, stations, features):
    models_log_dict = {'Station': [], 'Model': []}
    for key in config['DEFAULT'].keys():
        models_log_dict[key] = []
    models_log_dict.update({'R-squared': [], 'MAE': [], 'RMSE': [], 'Global radiation': [], 'Check-ins week ago': [],
                            'Check-outs week ago': []})

    for i, model in enumerate(models):
        models_log_dict['Station'].append(stations[i])
        models_log_dict['Model'].append(model[0])

        models_log_dict['UseNormalizedVisitors'].append(config_use_normalized_visitors)
        models_log_dict['UseEventStationDistance'].append(config_use_event_station_distance)
        models_log_dict['IncludeInstagramEvents'].append(config_include_instagram_events)
        models_log_dict['IncludeTicketmasterEvents'].append(config_include_ticketmaster_events)
        models_log_dict['UseTimeOfEvents'].append(config_use_time_of_events)
        models_log_dict['MaxHoursBeforeEvent'].append(int(config_max_hours_before_event))
        models_log_dict['MaxMinutesBeforeEvent'].append(int(config_max_minutes_before_event))
        models_log_dict['MaxHoursAfterEvent'].append(int(config_max_hours_after_event))
        models_log_dict['MaxMinutesAfterEvent'].append(int(config_max_minutes_after_event))
        models_log_dict['UseCOVIDStringency'].append(config_use_covid_stringency)
        models_log_dict['UseCOVIDMeasures'].append(config_use_covid_measures)
        models_log_dict['UseCOVIDCases'].append(config_use_covid_cases)
        models_log_dict['UseCOVIDDeaths'].append(config_use_covid_deaths)
        models_log_dict['Global radiation'].append('global_radiation' in features)
        models_log_dict['Check-ins week ago'].append('check-ins_week_ago' in features)
        models_log_dict['Check-outs week ago'].append('check-outs_week_ago' in features)

        models_log_dict['R-squared'].append(builtins.round(model[1], 3))
        models_log_dict['MAE'].append(builtins.round(model[2], 3))
        models_log_dict['RMSE'].append(builtins.round(model[3], 3))

    models_log_df = pd.DataFrame(models_log_dict)

    if os.path.exists('output/models_log.csv'):
        old_models_log_df = pd.read_csv('output/models_log.csv')
        models_log_df = pd.concat([old_models_log_df, models_log_df])
        models_log_df.drop_duplicates(
            subset=['Station', 'Global radiation', 'Check-ins week ago', 'Check-outs week ago'] + list(
                config['DEFAULT'].keys()), inplace=True)

    models_log_df.to_csv('output/models_log.csv', index=False)
