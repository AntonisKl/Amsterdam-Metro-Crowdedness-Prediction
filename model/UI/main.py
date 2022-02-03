import glob
import json
import os
from datetime import date, datetime, timedelta

import dash
import dash_leaflet as dl
import diskcache
import pandas as pd
import plotly.express as px
import requests
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.long_callback import DiskcacheLongCallbackManager
from dash_extensions.javascript import Namespace

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Global variables
marker_name = ''
complete = ''

# Set dates in date picker
today = datetime.today().strftime('%Y-%m-%d')
today_dt = today.split('-')

week_ahead = str(datetime.today() + timedelta(days=7)).split(' ')[0].split('-')

# Load map templates
url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '

# Read static geojson for metrolines and stations
df = json.load(open('metrolines.geojson'))

# Create geojson  
ns = Namespace("myNamespace", "mySubNamespace")
geojson = dl.GeoJSON(data=df)

# Create figure
fig_with_marker = px.line(
)

# Update Layout
fig_with_marker.update_layout(
    paper_bgcolor='rgb(49,48,47)',
    plot_bgcolor='rgb(49,48,47)',
    hovermode='closest',
    margin={'l': 30, 'b': 30, 't': 10, 'r': 0},
    font_color='white'
)

fig_with_marker.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)
fig_with_marker.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)

fig_with_marker.add_shape(  # add a horizontal "target" line
    type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
    x0=0, x1=1, xref="paper", y0=1500, y1=1500, yref="y"
)

# Markers

markers_ls = [
    dl.Marker(
        id='bijlmer',
        # title = 'Bijlmer',
        position=[52.312132, 4.947191],
        children=[
            dl.Tooltip('Station Bijlmer Arena')
        ]
    ),
    dl.Marker(
        id='centraal',
        # title = 'Centraal',
        position=[52.378027, 4.899773],
        children=[
            dl.Tooltip('Centraal Station')
        ]
    ),
    dl.Marker(
        id='zuid',
        # title = 'Zuid',
        position=[52.340165, 4.873144],
        children=[
            dl.Tooltip('Station Zuid')
        ]
    ),
]

cluster = dl.MarkerClusterGroup(id="markers_ls", children=markers_ls)

# Stylesheets

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the app.  
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, long_callback_manager=long_callback_manager)
app.title = 'GVB Predictions'
app.layout = html.Div([
    html.Div(
        [
            html.Div(
                [
                    html.Img(src='./assets/logo1.png', className='img-logo'),
                    html.H2(
                        'GVB PREDICTIONS'
                    ),
                    html.Div([
                        html.Div([
                            html.P('Features', className='dropdown-header'),
                            dcc.Dropdown(
                                options=[
                                    {'label': 'Year', 'value': 'year'},
                                    {'label': 'Month', 'value': 'month'},
                                    {'label': 'Weekday', 'value': 'weekday'},
                                    {'label': 'Holiday', 'value': 'holiday'},
                                    {'label': 'Vacation', 'value': 'vacation'},
                                    {'label': 'Temperature', 'value': 'temperature'},
                                    {'label': 'Wind Speed', 'value': 'wind_speed'},
                                    {'label': 'Precipitation', 'value': 'precipitation_h'},
                                    {'label': 'Global Radiation', 'value': 'global_radiation'},
                                    {'label': 'Check-ins week ago', 'value': 'check-ins_week_ago'},
                                    {'label': 'Check-outs week ago', 'value': 'check-outs_week_ago'},
                                ],
                                value=['year', 'month', 'weekday', 'holiday', 'vacation',
                                       'temperature', 'wind_speed', 'precipitation_h', 'global_radiation',
                                       'check-ins_week_ago', 'check-outs_week_ago'],
                                multi=True,
                                className='dropdown-class',
                                id='features-dropdown'
                            ),
                            html.Div([
                                html.P(
                                    'Note that features - Planned Event and Hour are crucial for running the modelling process and hence, cannot be omitted in the training.',
                                    style={
                                        'padding-top': '3%'
                                    })
                            ]),
                            html.Div([
                                html.P('Events improvements', className='dropdown-header'),
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Include Instagram events', 'value': 'includeInstagramEvents:true'},
                                        {'label': 'Include Ticketmaster events',
                                         'value': 'includeTicketmasterEvents:true'},
                                        {'label': 'Use events\' attendance', 'value': 'useNormalizedVisitors:true'},
                                        {'label': 'Affect only 3 hours before and after the events',
                                         'value': 'useTimeOfEvents:true'},
                                        # after event
                                        {'label': 'Use events\' location', 'value': 'useEventStationDistance:true'}
                                    ],
                                    value=[],
                                    multi=True,
                                    className='dropdown-class',
                                    id='events-dropdown'
                                )
                            ]),
                            html.Div([
                                html.P('COVID-19 datasets', className='dropdown-header'),
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Stringency', 'value': 'useCOVIDStringency:true'},
                                        {'label': 'Measures', 'value': 'useCOVIDMeasures:true'},
                                        {'label': 'Cases', 'value': 'useCOVIDCases:true'},
                                        {'label': 'Deaths', 'value': 'useCOVIDDeaths:true'}
                                    ],
                                    value=[],
                                    multi=True,
                                    className='dropdown-class',
                                    id='covid-dropdown'
                                )
                            ]),
                            html.Div([
                                html.Button('SUBMIT', id='submit-button', n_clicks=0),
                                html.Button('Recommended Model', id='rec-button', style={'float': 'right'}, n_clicks=0)
                            ],
                                className='submit-btn'
                            ),
                        ], )
                    ],
                        className='row'
                    )
                ],

                className='four columns div-user-controls'
            ),
            html.Div([
                html.Div([
                    dl.Map(
                        center=(52.3527598, 4.8936041),
                        zoom=11,
                        id='map',
                        style={'height': '45vh'},
                        children=[
                            dl.TileLayer(
                                url=url,
                                maxZoom=20,
                                attribution=attribution
                            ),
                            geojson,
                            cluster
                        ],
                    ),
                ]),
                html.Div([
                    'Select a station to update the following graph.'
                ],
                    className='text-padding station-title',
                    id='station-div'
                ),
                html.Div([
                    dcc.DatePickerSingle(
                        id='date-picker-single',
                        min_date_allowed=date(int(today_dt[0]), int(today_dt[1]), int(today_dt[2])),
                        max_date_allowed=date(int(week_ahead[0]), int(week_ahead[1]), int(week_ahead[2])),
                        initial_visible_month=date(int(today_dt[0]), int(today_dt[1]), int(today_dt[2])),
                        date=date(int(today_dt[0]), int(today_dt[1]), int(today_dt[2])),
                        display_format='DD/MM/YYYY'
                    ),
                ],
                    className='date-div',
                    id='date-picker-div'
                ),
                html.Div([
                    dcc.Loading(
                        id="loading-1",
                        type="circle",
                        children=dcc.Graph(
                            figure=fig_with_marker,
                            style={'height': '40vh'},
                            id='line-graph'
                        )
                    ),
                ],
                    id="line-graph-div"
                )
            ],
                id='map-div',
                className='eight columns div-for-charts bg-grey'
            )
        ],
        id="header",
        className='row',
    ),
])


@app.callback([Output("station-div", "children"),
               Output('line-graph', 'figure')],
              [Input(marker.id, "n_clicks") for marker in markers_ls],
              Input('date-picker-single', 'date'),
              Input('submit-button', 'n_clicks'),
              Input('features-dropdown', 'value'),
              Input('events-dropdown', 'value'),
              Input('covid-dropdown', 'value')
              )
def marker_click(one, two, three, date_value, submit, features, events, covid):
    global marker_name
    global complete

    if dash.callback_context.triggered[0]["prop_id"].split(".")[0] in ('centraal', 'zuid', 'bijlmer'):
        marker_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

        if marker_id == 'zuid':
            marker_name = 'Station Zuid'
        elif marker_id == 'bijlmer':
            marker_name = 'Station Bijlmer ArenA'
        elif marker_id == 'centraal':
            marker_name = 'Centraal Station'
        else:
            marker_name = 'Select a station to update the following graph.'

        if marker_name != 'Select a station to update the following graph.':
            selected_output = getlatestfile(marker_name, date_value)

            fig = setgraphinfo(selected_output, marker_name)

    elif dash.callback_context.triggered[0]["prop_id"].split(".")[0] == 'date-picker-single':

        selected_output = getlatestfile(marker_name, date_value)

        fig = setgraphinfo(selected_output, marker_name)

    elif dash.callback_context.triggered[0]["prop_id"].split(".")[0] == 'submit-button':
        result = getpredictions(features, events, covid)

        if result:
            complete = 'Success'
            selected_output = getlatestfile(marker_name, date_value)
            fig = setgraphinfo(selected_output, marker_name)
        else:
            complete = 'Error'
    return marker_name, fig


@app.callback([Output('features-dropdown', 'value'),
               Output('covid-dropdown', 'value'),
               Output('events-dropdown', 'value')],
              [Input(marker.id, "n_clicks") for marker in markers_ls],
              Input('date-picker-single', 'date')
              )
def update_features(one, two, three, date_value):
    features_values = []
    covid_values = []
    event_values = []
    global marker_name

    selected_output = None
    if dash.callback_context.triggered[0]["prop_id"].split(".")[0] in ('centraal', 'zuid', 'bijlmer'):
        marker_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

        if marker_id == 'zuid':
            marker_name = 'Station Zuid'
        elif marker_id == 'bijlmer':
            marker_name = 'Station Bijlmer ArenA'
        elif marker_id == 'centraal':
            marker_name = 'Centraal Station'
        else:
            marker_name = 'Select a station to update the following graph.'

        if marker_name != 'Select a station to update the following graph.':
            selected_output = getlatestfile(marker_name, date_value)
            event_values = read_config()

    elif dash.callback_context.triggered[0]["prop_id"].split(".")[0] == 'date-picker-single':
        selected_output = getlatestfile(marker_name, date_value)
        event_values = read_config()

    if selected_output is None:
        return features_values, covid_values, event_values

    # Features dropdown
    for col in selected_output.columns:
        if col in ('year', 'month', 'weekday', 'holiday', 'vacation',
                   'temperature', 'wind_speed', 'precipitation_h', 'global_radiation',
                   'check-ins_week_ago', 'check-outs_week_ago'):
            features_values.append(col)
        if col in ('stringency', 'cases', 'deaths', 'AdaptationOfWorkplace'):
            if col == 'stringency':
                covid_values.append('useCOVIDStringency:true')
            elif col == 'cases':
                covid_values.append('useCOVIDCases:true')
            elif col == 'deaths':
                covid_values.append('useCOVIDDeaths:true')
            else:
                covid_values.append('useCOVIDMeasures:true')

    return features_values, covid_values, event_values


def read_config():
    selected_values = []

    with open('../output/config.json', 'r') as f:
        data = json.load(f)

    if data['UseNormalizedVisitors'] == 1:
        selected_values.append('useNormalizedVisitors:true')
    if data['UseEventStationDistance'] == True:
        selected_values.append('useEventStationDistance:true')
    if data['IncludeInstagramEvents'] == 1:
        selected_values.append('includeInstagramEvents:true')
    if data['IncludeTicketmasterEvents'] == 1:
        selected_values.append('includeTicketmasterEvents:true')
    if data['UseTimeOfEvents'] == 1:
        selected_values.append('useTimeOfEvents:true')

    return selected_values


def getlatestfile(marker, date):
    # Get latest file from the folder

    list_of_files = glob.glob('../output/' + marker + '/*')
    latest_file = max(list_of_files, key=os.path.getctime)

    # Read latest file
    selected_output = pd.read_csv(latest_file)

    selected_output['sumcheckinsouts'] = (
            selected_output['check-ins_predicted'] + selected_output['check-outs_predicted'])

    selected_output = selected_output.loc[selected_output['datetime'] == date]

    return selected_output


def setgraphinfo(selected, marker):
    figure = px.line(
        selected,
        x="hour",
        y="sumcheckinsouts",
        # Markers Property
        markers=True,
    )

    figure.update_layout(
        paper_bgcolor='rgb(49,48,47)',
        plot_bgcolor='rgb(49,48,47)',
        hovermode='closest',
        margin={'l': 30, 'b': 30, 't': 10, 'r': 0},
        font_color='white',
        xaxis_title='Hour',
        yaxis_title='Total Check-ins and Check-outs'
    )

    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)
    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)

    if marker_name == 'Centraal Station':
        figure.add_shape(  # add a horizontal "target" line
            type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
            x0=0, x1=1, xref="paper", y0=8845, y1=8845, yref="y"
        )
    elif marker_name == 'Station Zuid':
        figure.add_shape(  # add a horizontal "target" line
            type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
            x0=0, x1=1, xref="paper", y0=5428, y1=5428, yref="y"
        )
    elif marker_name == 'Station Bijlmer ArenA':
        figure.add_shape(  # add a horizontal "target" line
            type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
            x0=0, x1=1, xref="paper", y0=3562, y1=3562, yref="y"
        )

    return figure


def getpredictions(features, events, covid):
    eventDict = {
        'includeInstagramEvents': 'false',
        'includeTicketmasterEvents': 'false',
        'useTimeOfEvents': 'false',
        'useNormalizedVisitors': 'false',
        'useEventStationDistance': 'false'
    }

    covidDict = {
        'useCOVIDStringency': 'false',
        'useCOVIDMeasures': 'false',
        'useCOVIDCases': 'false',
        'useCOVIDDeaths': 'false'
    }
    # GET Request
    URL = 'http://127.0.0.1:5000/train-and-predict'

    features = ','.join(features)
    features = features + ',hour,planned_event'

    if not events:
        events = eventDict
    elif len(events) < len(eventDict):
        events = dict(s.split(':') for s in events)
        for key in eventDict:
            if key not in events.keys():
                events[key] = 'false'
    else:
        events = dict(s.split(':') for s in events)

    if not covid:
        covid = covidDict
    elif len(covid) < len(covidDict):
        covid = dict(s.split(':') for s in covid)
        for key in covidDict:
            if key not in covid.keys():
                covid[key] = 'false'
    else:
        covid = dict(s.split(':') for s in covid)

    for key in covid:
        if key in ('useCOVIDStringency', 'useCOVIDCases', 'useCOVIDDeaths'):
            if covid[key] == 'true':
                features = features + ',' + key.split("COVID", 1)[1].lower().strip()

    PARAMS = {'features': features}
    PARAMS.update(events)
    PARAMS.update(covid)

    r = requests.get(url=URL, params=PARAMS)

    # Extracting data in json format
    data = r.json()

    return data['success']


if __name__ == '__main__':
    app.run_server()
