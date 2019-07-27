# Sentiment_twitter
#Sentiment analysis of tweets by companies ceo

import math
import time
symbols=" @#$%"
beyond = [ord(s) for s in symbols if ord(s)>40]
print(beyond)

beyond1 = list(filter(lambda c:c > 40,map(ord,symbols)))
print(beyond1)


start = time. time()
symbols="@#$s%"
beyond = [ord(s) for s in symbols if ord(s)>40]
print(beyond)
end = time. time()
print(end - start)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
from datetime import datetime

consumer_secret = "m2mRm7xH8ti2v5Uax81RZSKq07qFLx4X93SaHb3nNWebvMXdsz"
consumer_key = "aeZnM0UCukSwaPy8RNKcsMMvh"

access_token = "1661967871-PFSQD0Hatf3YzFes7O3dIJ8mNuXwZnIfsNNVv10"
access_token_secret = "n0DKLo7N6nPr4AB00x7lHxJtVjZfBxCrZiwjBTRP9u5TF"

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

target_users = ["@tim_cook","@JeffBezos","@elonmusk"]
sentiments = []

for target in target_users:
    print(target)
    counter = 0
    
    compound_list =[]
    Positive_list = []
    negative_list = []
    neutral_list = []
    
    public_tweets = api.user_timeline(target, count = 200)
    
    #Loop through each tweet
    for tweet in public_tweets:
        
        results = analyzer.polarity_scores(tweet["text"])
        compound = results["compound"]
        pos = results["pos"]
        neu = results["neu"]
        neg = results["neg"]
        tweets_ago = counter
        
        sentiments.append({"Tweet":tweet["text"],"CCeo":target,"Tweets Ago": counter,"Date":tweet["created_at"],"Compound":compound,"positive":pos,"Negative":neg,"Neutral":neu})
        counter = counter + 1
        
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd = sentiments_pd [['CCeo','Tweets Ago','Date','Tweet','Compound','positive','Neutral','Negative']]

df = sentiments_pd.to_csv("CCeo.csv",encoding= "utf-8",index = False)

orgs_colors_dict = {'@JeffBezos': 'blue','@tim_cook':'gold','@elonmusk': 'green'}



plt.scatter(sentiments_pd.groupby(["CCeo"]).get_group("@JeffBezos")["Tweets Ago"],
                sentiments_pd.groupby(["CCeo"]).get_group("@JeffBezos")["Compound"],
                  facecolors=orgs_colors_dict['@JeffBezos'], edgecolors='black', label="Jeff Bezos")

plt.scatter(sentiments_pd.groupby(["CCeo"]).get_group("@tim_cook")["Tweets Ago"],
                sentiments_pd.groupby(["CCeo"]).get_group("@tim_cook")["Compound"],
                  facecolors=orgs_colors_dict['@tim_cook'], edgecolors='black', label="Tim cook")

plt.scatter(sentiments_pd.groupby(["CCeo"]).get_group("@elonmusk")["Tweets Ago"],
                sentiments_pd.groupby(["CCeo"]).get_group("@elonmusk")["Compound"],
                  facecolors=orgs_colors_dict['@elonmusk'], edgecolors='black', label="Elon")



now = datetime.now()
now = now.strftime("%m/%d/%y")
plt.title(f'Sentiment Analysis of Elon Tweets ({now})')
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")

plt.xlim(100, 0)
plt.ylim(-1.0, 1.0)
yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
plt.yticks(yticks)

plt.legend(title="CCeo", bbox_to_anchor=(1, 1), frameon=False)

plt.savefig("sentiment_analysis_of_CCeo_tweets.png")
plt.show()

x_axis = np.arange(sentiments_pd["CCeo"].nunique())
tick_locations = [value+0.4 for value in x_axis]

plt.title(f'Overall CCeo Sentiment based on Twitter ({now})')
plt.xlabel("CCeo")
plt.ylabel("Tweet Polarity")

plt.bar(x_axis, sentiments_pd.groupby("CCeo").mean()["Compound"],
       color=orgs_colors_dict.values(), align="edge", width =1)
plt.xticks(tick_locations, sentiments_pd["CCeo"].unique())

plt.savefig("Bar_Sentiment.png")
plt.show()

from wordcloud import WordCloud, STOPWORDS 
from subprocess import check_output

df = pd.read_csv("CCeo.csv") 
df.head()

import matplotlib as mpl

mpl.rcParams['font.size']=12                
mpl.rcParams['savefig.dpi']=100             
mpl.rcParams['figure.subplot.bottom']=.1 

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['Tweet']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=1000)











