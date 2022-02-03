# Amsterdam Metro Crowdedness Prediction

The aim of this full-stack project is to predict and visualize crowdedness for 1 week ahead in 3 metro stations of
Amsterdam: Centraal Station, Station Zuid and Station Bijlmer ArenA. Except for the number of check-ins & check-outs for
each station, external factors are considered such as weather, events, holidays, vacations and COVID-19 pandemic.

## Description

The project consists of the following components:

- `instagram-event-scraper` &#8594; scraper for events from Instagram using instagram's public URLs
- `ticketmaster-event-fetcher` &#8594; fetcher for events from Ticketmaster API
- `model` &#8594; back-end and front-end for making predictions
    - `data_utils.py` &#8594; helper functions for data manipulation and logging
    - `model_utils.py` &#8594; functions for model pipeline
    - `predictions.ipynb` &#8594; notebook for running model pipeline
    - `predictions_server.py` &#8594; Flask server for running model pipeline
    - `UI` &#8594; front-end for running model pipeline

## Model Pipeline

1. Read and preprocess data
2. Merge data of external factors (e.g. weather) with check-ins & check-outs per hour
3. Interpolate missing check-ins & check-outs by using Random Forest algorithm
4. Split dataset into training, validation and test set
5. Create a separate Random Forest model for each of the 3 metro stations
6. Train each model with historical data (X)
7. Predict the check-ins & check-outs for each hour for 1 week ahead (Y)

## Getting Started

### Dependencies

* Python 3.7+
* All the libraries included in `requirements.txt`

### Installing

* Run `pip install -r requirements.txt`
* Datasets for **check-ins & check-outs** (`model/data/gvb/` & `model/data/gvb-herkomst/`), **
  weather** (`model/data/knmi/`) and **events** (`model/data/events/`) are expected to be in `model/` as per this
  directory structure:

```
model
└───data
    └───gvb
    │   └───<year>
    │   │   └───<month_number>
    │   │   │   └───<day_number>
    │   │   │       │   <csv_or_json.gz>
    │   │   │       │   ...
    │   │   │
    │   │   └───...
    │   └───...
    └───gvb-herkomst
    │   └───<year>
    │   │   └───<month_number>
    │   │   │   └───<day_number>
    │   │   │       │   <csv_or_json.gz>
    │   │   │       │   ...
    │   │   │
    │   │   └───...
    │   └───...
    └───knmi
    │   └───knmi
    │   │   └───<year>
    │   │   │   └───<month_number>
    │   │   │   │   └───<day_number>
    │   │   │   │       │   <json>
    │   │   │   │       │   ...
    │   │   │   │
    │   │   │   └───...
    │   │   └───...
    │   └───knmi-observations
    │       └───<year>
    │       │   └───<month_number>
    │       │   │   └───<day_number>
    │       │   │       │   <json>
    │       │   │       │   ...
    │       │   │
    │       │   └───...
    │       └───...
    └───events
        │   events_zuidoost.xlsx
        │
        └───instagram
        │   │   <csv>
        │   │   ...
        │
        └───ticketmaster
            │   <csv>
            │   ...
```

* **WARNING**: For the model to produce valid predictions, **check-ins & check-outs** (`model/data/gvb/`
  & `model/data/gvb-herkomst/`) and weather data (`model/data/knmi/`) should be manually up-to-date

### Executing programs

#### instagram-event-scraper

* Modify `usernames` array in `scraper.py` to include the usernames of the accounts which you want to be scraped
* Go to `instagram-event-scraper/` and run `python scraper.py`
* After execution, `instagram-event-scraper/events.csv` will be updated with the scraped events

#### ticketmaster-event-fetcher

* Create `ticketmaster-event-fetcher/config.py` containing `api_key=EXAMPLE` where `EXAMPLE` is a placeholder for your
  Ticketmaster API key
* Modify `year_to_fetch` variable in `fetcher.py` to fetch events for the year of your choice
* Go to `ticketmaster-event-fetcher/` and run `python fetcher.py`
* After execution, a file with format `ticketmaster-event-fetcher/events_amsterdam_center_DATE_TIME_UTC.csv` will be
  created with the fetched events

#### model

* Using `model/predictions.ipynb`:
    * Modify `config.ini` for the model to use the feature configuration of your choice
    * Run `model/predictions.ipynb`
    * See below bullet point "After execution"
* Using front-end and back-end server:
    * Go to `model/`, run `python predictions_server.py` and wait for the server output to show "Preprocessing finished"
      and be up
    * Go to `model/UI/`, run `python test.py` and wait for the front-end server to be up
    * Open the URL of the front-end server on a browser
    * Choose your desired parameters for the model and press "Submit"
    * After execution
        * If you press to any of the 3 available metro stations in the map, the graph should be updated with the current
          predictions
        * **Each station's folder** in `model/output/` will be updated with a new file with
          format `prediction_next_week_CURRENT-DATE.csv` which will contain the current predictions
        * **NOTE**: Only if you ran the model using `model/predictions.ipynb` notebook,
          then `model/output/models_log.csv`
          will be updated with the model's parameters and metrics

## Acknowledgments

* [instagram-scraper](https://github.com/arc298/instagram-scraper)