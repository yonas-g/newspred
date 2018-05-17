import datetime
import tweepy
import json
import time

def limit_handled(cursor):
    counter = 0
    while True:
        try:
            #time.sleep(3)
            counter+=1
            #print(counter)
            yield cursor.next()
        except tweepy.TweepError as e:
            print(e)
            time.sleep(15*60)

def save_to_csv(self, tweet):
    with open(r'saved_tweets_tag.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(list(tweet.values()))


with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)

auth = tweepy.OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
'''
api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
'''
tweets = []
api = tweepy.API(auth)
targetDate =   datetime.datetime(2013, 11, 5, 0, 0, 0)
query = 'americanair AND lang:en'
max_tweets = 100
timeout = 15
searched_tweets = [status for status in limit_handled(tweepy.Cursor(api.search, q=query).items(max_tweets))]
prev_res_last = searched_tweets[-1]
print('completed first query')
for tweet in searched_tweets:
    tweets.append(tweet._json)
while True:
    max_tweets = 100
    searched_tweets = [status for status in limit_handled(tweepy.Cursor(api.search, q=query, max_id=prev_res_last.id).items(max_tweets))]
    for tweet in searched_tweets:
        tweets.append(tweet._json)
    print(len(tweets))
    if len(searched_tweets) > 0:
        prev_res_last = searched_tweets[-1]
        timeout = 15
        print(targetDate, prev_res_last.created_at)
        if prev_res_last.created_at < targetDate:
            break
    else:
        print('no returns')
        if timeout == 0:
            break
        else:
            timeout-=1
        pass
with open('data/AALTweets.txt', 'w') as outfile:
    json.dump(tweets, outfile)
