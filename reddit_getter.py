from IPython import display
import sys
import time
import math
from pprint import pprint
import pandas as pd
import numpy as np
import praw
import json
from urllib.request import Request, urlopen
import urllib.error

reddit = praw.Reddit(client_id='//',
                     client_secret='//',
                     user_agent='//')

headlines = set()
now = time.time()
start = 1199145600
ts = start+86400
subreddit='politics'
from_=0
while ts < now-1000:

    url = 'https://api.pushshift.io/reddit/submission/search/?subreddit='+ str(subreddit) +'&sort=asc&size=999&after='+str(start)+'&before='+str(ts)
    print(url)
    q = Request(url)
    try:
        contents = urlopen(q).read()
        contents_json = json.loads(contents.decode('utf8'))
    except urllib.error.HTTPError as err:
        print('Encountered error:', err.code)
        print(err.msg)
        raise

    for submission in contents_json['data']:
        headlines.add((submission['title'], submission['created_utc']))
    start=ts
    ts=ts+86400
    print(len(headlines))
