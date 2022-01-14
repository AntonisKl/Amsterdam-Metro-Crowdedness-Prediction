import json
from datetime import datetime
import re
import locale
import subprocess
import os

import pandas as pd


def find_datetime_in_text(text, year=None):
    date_matches = re.findall(
        '(?i)([0-9][0-9][\n\r\s]+(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december|jan|feb|mrt|apr|mei|juni|juli|aug|sep|okt|nov|dec))',
        text)
    time_matches = re.findall('[0-1][0-9][:.][0-5][0-9]', text)
    if not date_matches or not time_matches:
        return None

    date_s = '{} {}'.format(date_matches[0][0], year if year else '2021')
    time_s = time_matches[0]
    datetime_s = '{} {}'.format(date_s, time_s)

    month_placeholder = '%b'
    if re.search('(?i)(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)',
                 datetime_s):
        month_placeholder = '%B'

    return datetime.strptime(datetime_s,
                             '%d {} %Y %H{}%M'.format(month_placeholder, time_s[2]))


class Event:
    def __init__(self, post_json, venue):
        self.venue = venue
        self.likes = post_json['edge_media_preview_like']['count']
        self.comments = post_json['edge_media_to_comment']['count']
        self.date_uploaded = datetime.fromtimestamp(post_json['taken_at_timestamp'])
        self.caption = post_json['edge_media_to_caption']['edges'][0]['node']['text']
        self.event_date = find_datetime_in_text(self.caption, str(self.date_uploaded.year))

    def __str__(self):
        return 'Likes: {}, Date uploaded: {}, Event date: {}\nCaption: {}'.format(self.likes, self.date_uploaded,
                                                                                  self.event_date, self.caption)


def main():
    usernames = [
        'paradisoadam',  # already scraped
        # 'johancruijffarena', # very low number of events (2 per 500 posts)
        # 'ziggodome',  # already scraped
        # 'afaslive', # already scraped
        # 'ibisamsterdamstopera', # no recent events
        # 'concertgebouw',    # already scraped
        # 'beursvanberlageofficial', # already scraped
        # 'theatercarre', # already scraped
        # 'melkwegamsterdam', # already scraped
        # 'olympischstadion', # already scaped
        # 'raiamsterdam' # no recent events
    ]

    with open('usernames.txt', mode='wt', encoding='utf-8') as file:
        file.write('\n'.join(usernames))
        file.write('\n')

    print('Running scraper...')
    subprocess.run(['scrape_posts.bat'])

    locale.setlocale(locale.LC_TIME, 'nl_NL')

    output_csv = 'events.csv'
    write_header = not os.path.exists(output_csv) or (os.path.exists(output_csv) and os.stat(output_csv).st_size == 0)
    events_df = None
    usernames_f = open('usernames.txt', 'r', encoding='utf8')
    for username in usernames_f.readlines():
        username = username.strip()
        f = open('{}/{}.json'.format(username, username), encoding='utf8')
        data = json.load(f)

        events = [Event(item, username) for item in data['GraphImages'] if
                  item['edge_media_to_caption']['edges'] and find_datetime_in_text(
                      item['edge_media_to_caption']['edges'][0]['node']['text'])]

        if not events:
            break

        if events_df is None:
            events_df = pd.DataFrame(columns=events[0].__dict__.keys())
        print(events[0])

        for event in events:
            events_df = events_df.append(event.__dict__, ignore_index=True)
        print(events_df.head())

    if events_df is not None:
        events_df.to_csv(output_csv, mode='a', header=write_header)


if __name__ == '__main__':
    main()
