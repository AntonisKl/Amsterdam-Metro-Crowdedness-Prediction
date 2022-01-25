import requests
import dash  
from dash import html 
from dash import dcc
import dash_leaflet as dl  
import dash_leaflet.express as dlx  
from dash_extensions.javascript import Namespace
from dash_extensions.javascript import assign
import geopandas as gpd
import json
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from datetime import date, datetime, timedelta
import glob
import os
import time
from dash.long_callback import DiskcacheLongCallbackManager
import dash_loading_spinners as dls
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)


#Global variables
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

#Update Layout
fig_with_marker.update_layout(
    paper_bgcolor = 'rgb(49,48,47)',
    plot_bgcolor = 'rgb(49,48,47)',
    hovermode='closest',
    margin={'l': 30, 'b': 30, 't': 10, 'r': 0},
    font_color = 'white'
)


fig_with_marker.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)
fig_with_marker.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)


fig_with_marker.add_shape( # add a horizontal "target" line
    type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
    x0=0, x1=1, xref="paper", y0=1500, y1=1500, yref="y"
)


# Markers
 
markers_ls= [
    dl.Marker(
        id = 'bijlmer',
        # title = 'Bijlmer',
        position=[52.312132, 4.947191], 
        children = [
            dl.Tooltip('Station Bijlmer Arena')
        ]
    ),
    dl.Marker(
        id = 'centraal',
        # title = 'Centraal',
        position=[52.378027, 4.899773], 
        children = [
            dl.Tooltip('Centraal Station')
        ]
    ),
    dl.Marker(
        id = 'zuid',
        # title = 'Zuid',
        position=[52.340165, 4.873144], 
        children = [
        dl.Tooltip('Station Zuid')
        ]
    ),
]

cluster = dl.MarkerClusterGroup(id="markers_ls", children=markers_ls)

# Stylesheets

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the app.  
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, long_callback_manager=long_callback_manager)  
app.layout = html.Div([  
    html.Div(
            [
                html.Div(
                    [
                        html.Img(src='./assets/logo1.png', className='img-logo'),
                        html.H2(
                            'GVB PREDICTIONS'
                        ),
                        html.P('Select one or more features to see the updated predictions'),
                        html.Div([
                            html.Div([
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Year', 'value': 'year'},
                                        {'label': 'Month', 'value': 'month'},
                                        {'label': 'Weekday', 'value': 'weekday'},
                                       # {'label': 'Hour', 'value': 'hour', 'disabled': True},
                                        {'label': 'Holiday', 'value': 'holiday'},
                                        {'label': 'Vacation', 'value': 'vacation'},
                                       # {'label': 'Planned Event', 'value': 'planned_event'},
                                        {'label': 'Temperature', 'value': 'temperature'},
                                        {'label': 'Wind Speed', 'value': 'wind_speed'},
                                        {'label': 'Precipitation', 'value': 'precipitation_h'},
                                        {'label': 'Global Radiation', 'value': 'global_radiation'},
                                    ],
                                    value=['year', 'month', 'weekday', 'hour', 'holiday', 'vacation', 'planned_event',
                                    'temperature', 'wind_speed', 'precipitation_h',  'global_radiation'],
                                    multi=True,
                                    className = 'dropdown-class',
                                    id = 'features-dropdown'
                                ),
                                html.Div([
                                    html.P('EVENTS', className = 'dropdown-header'),
                                    dcc.Dropdown(
                                    options=[
                                        {'label': 'Instagram', 'value': 'includeInstagramEvents:true'},
                                        {'label': 'Ticketmaster', 'value': 'includeTicketmasterEvents:true'},
                                        {'label': 'Normalized Visitors', 'value': 'useNormalizedVisitors:true'},
                                        {'label': 'Time Of Events', 'value': 'useTimeOfEvents:true'}
                                    ],
                                    value=[],
                                    multi=True,
                                    className = 'dropdown-class',
                                    id = 'events-dropdown'
                                    )
                                ]),
                                html.Div([
                                    html.P('COVID-19', className = 'dropdown-header'),
                                    dcc.Dropdown(
                                    options=[
                                        {'label': 'Stringency', 'value': 'useCOVIDStringency:true'},
                                        {'label': 'Measures', 'value': 'useCOVIDMeasures:true'},
                                        {'label': 'Cases', 'value': 'useCOVIDCases:true'},
                                        {'label': 'Deaths', 'value': 'useCOVIDDeaths:true'}
                                    ],
                                    value=[],
                                    multi=True,
                                    className = 'dropdown-class',
                                    id = 'covid-dropdown'
                                    )
                                ]),
                                html.Div([
                                    html.Button('SUBMIT', id = 'submit-button', n_clicks=0),
                                    #dcc.Input(id="loading-input-1", value='Input triggers local spinner'),
                                    # dcc.Loading(
                                    #     id="loading-1",
                                    #     type="circle",
                                    #     isloading = True,
                                    #     children=html.Div(id="loading-output-1")
                                    # ),
                                    html.Div([
                                        dls.Roller(
                                            color= '#007eff',
                                            id = 'loading-component',
                                            #show_initially = False
                                        )
                                    ],
                                        id = 'loading-div',
                                        style = {
                                            'display':'none'
                                        }
                                    )
                                ],
                                    className='submit-btn'
                                )
                            ],)
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
                            children = [
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
                        id = 'station-div'
                    ),
                    html.Div([
                        dcc.DatePickerSingle(
                            id='date-picker-single',
                            min_date_allowed=date(int(today_dt[0]), int(today_dt[1]), int(today_dt[2])),
                            max_date_allowed=date(int(week_ahead[0]), int(week_ahead[1]), int(week_ahead[2])),
                            initial_visible_month=date(int(today_dt[0]), int(today_dt[1]), int(today_dt[2])),
                            date=date(int(today_dt[0]), int(today_dt[1]), int(today_dt[2])),
                            display_format = 'DD/MM/YYYY'
                        ),
                    ],
                        className = 'date-div',
                        id = 'date-picker-div'    
                    ),
                    html.Div([
                        dcc.Graph(
                            figure= fig_with_marker,
                            style={'height': '40vh'},
                            id = 'line-graph'
                        )
                    ],
                        id = "line-graph-div"
                    )  
                ],
                    id = 'map-div',
                    className = 'eight columns div-for-charts bg-grey'
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
              Input('events-dropdown','value'),
              Input('covid-dropdown', 'value')
              )
def marker_click(one, two, three, date_value, submit, features, events, covid):
    global marker_name
    global complete
    print(dash.callback_context.triggered[0]["prop_id"].split(".")[0]) 
    
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
            print(selected_output)
            fig = setgraphinfo(selected_output)
        
    elif dash.callback_context.triggered[0]["prop_id"].split(".")[0] == 'date-picker-single':
        
        selected_output = getlatestfile(marker_name, date_value)

        fig = setgraphinfo(selected_output)
    
    elif dash.callback_context.triggered[0]["prop_id"].split(".")[0] == 'submit-button':
        print(date_value)
        complete = 'Start'
        print(complete)
        result = getpredictions(features, events, covid)
        
        if result == True:
            complete = 'Success'
            selected_output = getlatestfile(marker_name, date_value)
            fig = setgraphinfo(selected_output)
        else:
            complete = 'Error'
    return marker_name, fig


# @app.callback(Output("loading-div", "style"), 
#             Input("submit-button", "n_clicks"),
#             prevent_initial_call=True)
# def input_triggers_spinner(value):
#     global complete

#     print(complete)

#     if complete == 'Start':
#         return {'display':'block'}
#     else:
#         return {'display': 'none'}  



def getlatestfile(marker, date):
    # Get latest file from the folder

    list_of_files = glob.glob('../output/' + marker + '/*')
    latest_file = max(list_of_files, key=os.path.getctime)

    #Read latest file
    selected_output = pd.read_csv(latest_file)
    selected_output['average_checkinouts'] = (selected_output['check-ins_predicted']+selected_output['check-outs_predicted'])

    selected_output = selected_output.loc[selected_output['datetime'] == date]
    
    return selected_output


def setgraphinfo(selected):
    figure = px.line(
            selected,
            x = "hour",
            y = "average_checkinouts",
            #Markers Property
            markers = True,
        )

    figure.update_layout(
        paper_bgcolor = 'rgb(49,48,47)',
        plot_bgcolor = 'rgb(49,48,47)',
        hovermode='closest',
        margin={'l': 30, 'b': 30, 't': 10, 'r': 0},
        font_color = 'white'
    )


    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)
    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)


    figure.add_shape( # add a horizontal "target" line
        type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
        x0=0, x1=1, xref="paper", y0=1500, y1=1500, yref="y"
    )

    return figure


def getpredictions(features, events, covid):
    # GET Request
    URL = 'http://127.0.0.1:5000/train-and-predict'
    
    features = ','.join(features)
    features = features+ ',hour,planned_event'

    print(features)


    events =  dict(s.split(':') for s in events)

    covid = dict(s.split(':') for s in covid)

    PARAMS = {'features': features}
    PARAMS.update(events)
    PARAMS.update(covid)
    
    print(PARAMS)
    r= requests.get(url = URL, params = PARAMS)
    
    # Extracting data in json format
    data = r.json()
    #return ''
    return data['success']


# @app.callback(
#     Output('line-graph', 'figure'),
#     [Input('submit-button', 'n_clicks'),
#     Input('features-dropdown', 'value')]
# )
# def update_output(n_clicks, value):
#     # GET Request
#     URL = 'http://127.0.0.1:5000/train-and-predict'

#     PARAMS = {'features': 'year,month,weekday,hour,holiday,vacation,planned_event,temperature,wind_speed,precipitation_h,global_radiation',
#     'useMeasures': 'true',
#     'useNormalizedVisitors': 'false',
#     'maxHoursBeforeEvent': '24'}

#     r= requests.get(url = URL, params = PARAMS)
#     print(r)

#     #  Extracting data in json format
#     data = r.json()
#     return data



# @app.long_callback(
#     output=Output("loading-div", "style"),
#     inputs=Input("submit-button", "n_clicks"),
#     running=[
#         (Output("submit-button", "disabled"), True, False),
#         # (
#         #     Output("progress_bar", "style"),
#         #     {"visibility": "visible"},
#         #     {"visibility": "hidden"},
#         # ),
#     ],
#     # progress=[Output("progress_bar", "value"), Output("progress_bar", "max")]
# )
# def callback(n_clicks):
#     total = 10
#     if n_clicks is not None:
#         for i in range(total):
#             time.sleep(0.5)
#     return [f"Clicked {n_clicks} times"]



# @app.callback(Output("line-graph", "figure"),
#               [Input('my-date-picker-single', 'date'),
#               Input('station-div', 'children' )])
# def update_output(date_value, value):
#     print(date_value)
#     print(value)

#     # Get latest file from the folder

#     list_of_files = glob.glob('../output/' + value + '/*')
#     latest_file = max(list_of_files, key=os.path.getctime)
#     print(latest_file)

#     #Read latest file
#     selected_output = pd.read_csv(latest_file)
#     selected_output['average_checkinouts'] = (selected_output['check-ins_predicted']+selected_output['check-outs_predicted'])/2
#     selected_output = selected_output.loc[selected_output['datetime'] == today]

#     figure = px.line(
#             selected_output,
#             x = "hour",
#             y = "average_checkinouts",
#             #Markers Property
#             markers = True,
#         )

#     figure.update_layout(
#         paper_bgcolor = 'rgb(49,48,47)',
#         plot_bgcolor = 'rgb(49,48,47)',
#         hovermode='closest',
#         margin={'l': 30, 'b': 30, 't': 10, 'r': 0},
#         font_color = 'white'
#     )


#     figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)
#     fig_with_marker.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(77,76,76)', zeroline=False)


#     figure.add_shape( # add a horizontal "target" line
#         type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
#         x0=0, x1=1, xref="paper", y0=1500, y1=1500, yref="y"
#     )

#     return figure


if __name__ == '__main__':  
    app.run_server(debug = True)
