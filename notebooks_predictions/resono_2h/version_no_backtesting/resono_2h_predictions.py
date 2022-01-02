## Resono 2h predictions - logic (from reading in from DB to writing predictions to DB)

''' Make 2h-ahead predictions of the visitor counts ('total_count' column in the 'ingested.resono' table) for all or a selection of Resono locations that are included in druktebeeld. 

Predictions are written to a new table 'public.resono_2h_pred_count' with the following additional columns: 
- 'total_count_predicted': predicted total counts(for the next 8 time slots per location) 
- 'data_version': version of the data (feature set)
- 'model_version': version of the model (type and settings)
- 'predicted_at': timestamp of prediction (moment prediction was made)
'''

# ### Preparations
# 
# Change directory to folder that contains the DB credentials/folder that contains the function files in code block below.

def install_packages():
    # (Re-)Installs packages.
    
    get_ipython().run_cell_magic('bash', '', 'pip install imblearn\npip install mord\npip install psycopg2-binary\npip install workalendar\npip install eli5\n pip install plotly')
    
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)


get_ipython().run_cell_magic('capture', '', 'install_packages()')

import os
import pandas as pd
os.chdir("/home/jovyan/Crowd-prediction/Credentials")
import env_az
os.chdir("/home/jovyan/gitops/central_storage_analyses/notebooks_predictions/resono_2h")
import resono_2h_predictions_helpers as resono_pred  # Helper functions

# ### Settings

# #### Arguments for functions

# frequency of sampling for data source to predict
freq = '15min'

# what period to predict for operational forecast (samples)
predict_period = 8  ## Two hours in 15-minute steps
# how many samples in a day
n_samples_day = 96
# how many samples in a week
n_samples_week = n_samples_day*7

# list of column name(s) of variabe to predict (can also be "all")
Y_names = "all" 
#Y_names = ["Albert Cuyp", "De Dam West", "De Dam Oost", "Kalverstraat Noord", "Kalverstraat Zuid",
          #"Vondelpark Oost 1", "Vondelpark Oost 2", "Vondelpark Oost 3", "Vondelpark West",
          #"Rembrandtplein", "Leidseplein", "Nieuwmarkt", "Buikslotermeerplein",
          #"Rembrandtpark", "Westerpark Centrum", "Westerpark West", "Westerpark Oost",
          #"Oosterpark", "Erasmuspark", "Flevopark",
          #"Park Frankendael", "Park Somerlust", "Bijlmerplein", "Waterlooplein", "Sarphatipark",
          #"Rokin", "Spui", "Damrak", "Nieuwendijk"]

# data source (for which the predictions are made)
data_source = 'resono'

# type of prediction (count -> regression or level -> classification)
target = 'count'

# input for model
use_smote = True

# input for starting of learnset 
start_learnset = resono_pred.get_start_learnset(train_length = 8, date_str = None)  # 8 weeks of training data

# perform outlier removal ("yes" or "no")
outlier_removal = "yes"

# set versions (for storing results)
current_model_version = 'lr_0_0'
current_data_version = "1_0" 


# ### Get predictions

# #### 1. Prepare data sets

base_df, resono_df, resono_df_raw, start_prediction, end_prediction, thresholds, Y_names_all = resono_pred.prepare_data(env_az, 
                                                                                                           freq, 
                                                                                                           predict_period, 
                                                                                                           n_samples_day, 
                                                                                                           Y_names, 
                                                                                                           target,
                                                                                                           start_learnset)


# #### 2. Make predictions and store in data frame

# Initialize data frame with predictions
final_df = pd.DataFrame()

# Predict for each location
for idx, Y in enumerate(Y_names_all):
    
    # Preprocessed data frame for this location
    preprocessed_df = resono_pred.get_location_df(base_df, resono_df, Y)
    
    # Gather predictons for this location
    prepared_df, predictions, y_scaler, thresholds_scaled_one = resono_pred.get_resono_predictions(preprocessed_df, resono_df_raw, freq, predict_period, n_samples_day, 
                                                             n_samples_week, Y, data_source, target, 
                                                             outlier_removal, start_learnset, use_smote,
                                                             current_model_version, current_data_version, 
                                                             start_prediction, end_prediction, thresholds)

    # Add predictions to final data frame
    final_df = pd.concat([final_df, predictions], 0)
    
###  Store data

if target == 'count':
    final_df.to_sql('resono_2h_pred_count', con = engine_azure, if_exists = 'append', index = False)
