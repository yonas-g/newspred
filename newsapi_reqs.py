from urllib.request import Request, urlopen
import urllib.error

import json

#return dict of answer
def get_top_headlines(api_key, params={}):

    url = 'https://newsapi.org/v2/top-headlines?'
    for key in params.keys():
        if not params[key]==None:
            url = url+str(key)+'='+str(params[key])+'&'
    url = url[0:-1]
    print(url)
    q = Request(url)
    q.add_header('x-api-key', api_key)

    contents = urlopen(q).read()

    contents_json = json.loads(contents.decode('utf8'))
    return contents_json


#return dict of answer
def get_everything(api_key, params={}):

    url = 'https://newsapi.org/v2/everything?'
    for key in params.keys():
        if not params[key]==None:
            url = url+str(key)+'='+str(params[key])+'&'
    url = url[0:-1]
    print(url)
    q = Request(url)
    q.add_header('x-api-key', api_key)
    try:
        contents = urlopen(q).read()
    except urllib.error.HTTPError as err:
        print('Encountered error:', err.code)
        print(err.msg)
        raise

    contents_json = json.loads(contents.decode('utf8'))
    return contents_json
