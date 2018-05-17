from IPython import display
import sys
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords


def process_text(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)

    return tokens



stop_words = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

reddit = praw.Reddit(client_id='Bb9leu_CyhDY2w',
                     client_secret='meX0awmURb8umr4OMsVjPrhmXGg',
                     user_agent='gniorg')

headlines = set()

for submission in reddit.subreddit('politics').new(limit=None, from='8fb2ks', to='8fb0w1'):
    print(submission.name)
    input()
    headlines.add(submission.title)
print(len(headlines))

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1

df2 = df[['headline', 'label']]
#df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)

print(df.label.value_counts(normalize=True) * 100)
'''fig, ax = plt.subplots(figsize=(8, 8))

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()'''

pos_lines = list(df[df.label == 1].headline)

pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)

neg_lines = list(df[df.label == -1].headline)

neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)

print(pos_freq.most_common(20))

'''y_val = [x[1] for x in pos_freq.most_common()]
y_val2 = [x[1] for x in neg_freq.most_common()]

fig = plt.figure(figsize=(10,5))
plt.plot(y_val, 'b')
plt.plot(y_val2, 'r')

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")
plt.show()'''
