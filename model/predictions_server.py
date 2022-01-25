from distutils.util import strtobool

from flask import Flask, jsonify, request
from flask_script import Manager, Server

import data_utils
from model_utils import read_data, preprocess_data, clean_data, split_data_for_modelling, model, predict_and_save

# These global variables are written only on preprocess_data() and can be written only once.
gvb_dfs_final, covid_df, covid_measures_df, covid_cases_deaths_df, holiday_df, vacations_df, knmi_forecast, events = None, None, None, None, None, None, None, None
covid_measures_columns = []


def preprocess():
    global covid_measures_df, covid_cases_deaths_df, vacations_df, events, covid_df, holiday_df, knmi_forecast, gvb_dfs_final, covid_measures_columns

    # Global variables are written only once
    if gvb_dfs_final is not None:
        return

    herkomst_2020, bestemming_2020, herkomst_2021, bestemming_2021, knmi_obs, knmi_preds, covid_measures_df, covid_cases_deaths_df, covid_df_raw, holidays_data_raw, vacations_df, events = read_data()
    covid_measures_columns = list(covid_measures_df.columns.values)

    gvb_dfs_merged, covid_df, holiday_df, knmi_forecast = preprocess_data(herkomst_2020, bestemming_2020, herkomst_2021,
                                                                          bestemming_2021, knmi_obs, knmi_preds,
                                                                          covid_measures_df, covid_cases_deaths_df,
                                                                          covid_df_raw,
                                                                          holidays_data_raw,
                                                                          vacations_df, events)

    gvb_dfs_final = clean_data(gvb_dfs_merged)

    print('Preprocessing finished')


class CustomServer(Server):
    def __call__(self, app, *args, **kwargs):
        preprocess()
        return Server.__call__(self, app, *args, **kwargs)


app = Flask(__name__)
manager = Manager(app)

manager.add_command('runserver', CustomServer())


@app.route('/train-and-predict')
def train_and_predict():
    features_s = request.args.get('features')
    # features are mandatory
    if not features_s:
        return 'No features given', 400
    features = features_s.split(',')

    config_use_normalized_visitors = request.args.get('useNormalizedVisitors', type=strtobool)
    config_include_instagram_events = request.args.get('includeInstagramEvents', type=strtobool)
    config_include_ticketmaster_events = request.args.get('includeTicketmasterEvents', type=strtobool)
    config_use_time_of_events = request.args.get('useTimeOfEvents', type=strtobool)
    config_max_hours_before_event = request.args.get('maxHoursBeforeEvent', type=int)
    config_max_minutes_before_event = request.args.get('maxMinutesBeforeEvent', type=int)
    config_use_covid_stringency = request.args.get('useCOVIDStringency', type=strtobool)
    config_use_covid_measures = request.args.get('useCOVIDMeasures', type=strtobool)
    config_use_covid_cases = request.args.get('useCOVIDCases', type=strtobool)
    config_use_covid_deaths = request.args.get('useCOVIDDeaths', type=strtobool)

    # Set configuration
    if config_use_normalized_visitors is not None:
        data_utils.config_use_normalized_visitors = config_use_normalized_visitors

    if config_include_instagram_events is not None:
        data_utils.config_include_instagram_events = config_include_instagram_events

    if config_include_ticketmaster_events is not None:
        data_utils.config_include_ticketmaster_events = config_include_ticketmaster_events

    if config_use_time_of_events is not None:
        data_utils.config_use_time_of_events = config_use_time_of_events

    if config_max_hours_before_event is not None:
        data_utils.config_max_hours_before_event = config_max_hours_before_event

    if config_max_minutes_before_event is not None:
        data_utils.config_max_minutes_before_event = config_max_minutes_before_event

    if config_use_covid_stringency is not None:
        data_utils.config_use_covid_stringency = config_use_covid_stringency

    if config_use_covid_measures is not None:
        data_utils.config_use_covid_measures = config_use_covid_measures
        if config_use_covid_measures:
            features += covid_measures_columns

    if config_use_covid_cases is not None:
        data_utils.config_use_covid_cases = config_use_covid_cases

    if config_use_covid_deaths is not None:
        data_utils.config_use_covid_deaths = config_use_covid_deaths

    # # features = ['year', 'month', 'weekday', 'hour', 'holiday', 'vacation', 'planned_event', 'temperature', 'wind_speed', 'precipitation_h','global_radiation'] + list(covid_measures_df.columns.values)
    # features = ['year', 'month', 'weekday', 'hour', 'holiday', 'vacation', 'planned_event', 'temperature', 'wind_speed',
    #             'precipitation_h', 'global_radiation', 'stringency']

    data_splits, X_train_splits, y_train_splits, X_validation_splits, y_validation_splits, X_test_splits, y_test_splits, X_predict_dfs = split_data_for_modelling(
        gvb_dfs_final, covid_df, covid_measures_df, covid_cases_deaths_df, holiday_df, vacations_df, knmi_forecast, events,
        features)

    final_models = model(data_splits, X_train_splits, y_train_splits, X_validation_splits, y_validation_splits,
                         X_test_splits, y_test_splits, features)

    predict_and_save(final_models, X_predict_dfs)

    return jsonify(success=True)


if __name__ == "__main__":
    manager.run()
