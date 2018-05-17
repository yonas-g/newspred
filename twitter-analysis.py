from collections import Counter
import pandas as pd
import ast
import csv
import json


#tweets = pd.read_csv("saved_tweets_tag.csv", header=None)
#tweets.columns=['user', 'user_loc', 'text','hashtags']

with open('data/AALTweets.txt') as file:
    tweets = json.load(file)



for tweet in tweets:
    print(tweet['created_at'])

print(len(tweets))

'''
print(tweets[:2])
# Extract hashtags and put them in a list
list_hashtag_strings = [entry for entry in tweets['hashtags']]
list_hashtag_lists = ast.literal_eval(','.join(list_hashtag_strings))
hashtag_list = [ht.lower() for list_ in list_hashtag_lists for ht in list_]

# Count most common hashtags
counter_hashtags = Counter(hashtag_list)
print(counter_hashtags.most_common(20))
'''
