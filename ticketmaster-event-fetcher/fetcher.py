import os
from datetime import datetime

import pandas as pd
import pytz
import requests

import config

year_to_fetch = 2022
url_events_amsterdam_center_2022 = 'https://app.ticketmaster.com/discovery/v2/events.json?size=200&geoPoint=u173z&radius=5&startDateTime={year}-01-01T00:00:00Z&endDateTime={year}-12-30T00:00:00Z&unit=km&sort=distance,date,asc&apikey={api_key}'.format(
    year=year_to_fetch, api_key=config.api_key)
output_csv = 'events_amsterdam_center_{}_{}_UTC.csv'.format(year_to_fetch,
                                                            datetime.now(pytz.utc).strftime("%d-%m-%Y_%H.%M.%S"))


class Event:
    def __init__(self, event_json):
        self.name = event_json['name']
        if event_json['_embedded']['venues'][0]['address']['line1'] == 'Lijnbaansgracht 238':
            self.venue = 'Lovelee'
        elif event_json['_embedded']['venues'][0]['address']['line1'] == 'Lijnbaansgracht 234 A':
            self.venue = 'Melkweg'
        else:
            self.venue = event_json['_embedded']['venues'][0]['name']
        self.datetime = event_json['dates']['start']['dateTime']

    def __str__(self):
        return 'Date: {}\nName: {}\nVenue: {}'.format(self.datetime, self.name,
                                                      self.venue)


def get_events_amsterdam_center_2022_page(page_id):
    return url_events_amsterdam_center_2022 + '&page=' + str(page_id)


def get_response_json(url):
    response = requests.get(url)
    response_json = response.json()

    if 'errors' in response_json:  # error encountered or last page reached
        return None

    return response_json


def main():
    response_json = get_response_json(get_events_amsterdam_center_2022_page(0))
    if response_json is None:
        print('Error encountered. Stopping execution...')
        exit()

    total_pages = response_json['page']['totalPages']

    write_header = not os.path.exists(output_csv) or (os.path.exists(output_csv) and os.stat(output_csv).st_size == 0)
    events_df = None
    for page_id in range(total_pages):
        response_json = get_response_json(get_events_amsterdam_center_2022_page(0))
        if response_json is None:
            print('Error encountered. Stopping execution...')
            exit()

        events_json = response_json['_embedded']['events']

        events = [Event(event_json) for event_json in events_json if len(event_json['_embedded']['venues']) > 0]

        if events_df is None:
            events_df = pd.DataFrame(columns=events[0].__dict__.keys())

        for event in events:
            events_df = events_df.append(event.__dict__, ignore_index=True)

    if events_df is not None:
        events_df.to_csv(output_csv, mode='a', header=write_header)


if __name__ == '__main__':
    main()
