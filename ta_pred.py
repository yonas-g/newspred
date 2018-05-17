from newsapi_reqs import *
from manage_db import *
import pickle
import sqlite3
from sys import exit

api_key='0b5671d9fd7740a98ce8b745b474185e'

db_news = 'data/news.db'
#if not os.path.isfile(db_news):
#create_db_news(db_news)

top_headlines_params = {'country': None, 'category': None, 'sources': None, 'q': None, 'pageSize':20, 'page':1}
everything_params = {'q':None, 'sources':None, 'domains':None, 'from':None, 'to':None, 'language':None, 'sortBy':None, 'pageSize':100, 'page':1}
top_headlines_params['q']='google'
top_headlines_params['country']='us'

everything_params['q']='google'
everything_params['from']='2018-04-01'
everything_params['language']='en'
everything_params['sortBy']='publishedAt'
everything_params['sources']='bloomberg,business-insider,cnn,google-news,the-new-york-times,the-washington-post,wired,reuters,reddit-r-all,cbc-news,bbc-news'

#contents_json = get_top_headlines(api_key, params=top_headlines_params)
try:
    contents_json=get_everything(api_key, params=everything_params)
except:
    print('exiting')
    exit(1)

#filename = 'data/trending_'+str(everything_params['q'])

#with open(filename, 'wb') as fid:
#    pickle.dump(contents_json, fid)

save_db_news(db_news, contents_json['articles'])

while(1):
    if contents_json['totalResults']>everything_params['pageSize']*everything_params['page']:
        everything_params['page'] = everything_params['page'] +1
        try:
            contents_json=get_everything(api_key, params=everything_params)
        except:
            print('exiting')
            exit(1)
        save_db_news(db_news, contents_json['articles'])
    else:
        break
    print(contents_json['totalResults']-everything_params['pageSize']*everything_params['page'])
print('finished!')
