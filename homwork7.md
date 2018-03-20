

```python
import tweepy
import time
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import json
import pandas as pd
from config import consumer_key,consumer_secret,access_token,access_token_secret
```


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
##Taken from class..still dont know how it works..
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

```


```python
target_user = ["@BBC","@CBS","@CNN","@FoxNews","@NYTimes"]
sentiment = []
```


```python
for each in target_user:
    count1 = 0
    
    tweets = api.user_timeline(each,count=100)
    
    for tweet in tweets:
        
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = count1
        tweet_text = tweet["text"] 
        sentiment.append({"User": each,
                         "Date": tweet["created_at"],
                         "Compound": compound,
                         "Positive": pos,
                         "Negative": neg,
                         "Neutral": neu,
                         "Tweets Ago": count1,
                         "Text": tweet_text})
        count1+=1


    
```


```python
sentiments_news = pd.DataFrame.from_dict(sentiment)
sentiments_news.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>Tue Mar 20 19:03:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Michael Portillo travels across India guided b...</td>
      <td>0</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.3859</td>
      <td>Tue Mar 20 18:33:01 +0000 2018</td>
      <td>0.160</td>
      <td>0.768</td>
      <td>0.071</td>
      <td>Can you watch this without laughing? 😹🔊 You'll...</td>
      <td>1</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3400</td>
      <td>Tue Mar 20 17:33:03 +0000 2018</td>
      <td>0.102</td>
      <td>0.738</td>
      <td>0.160</td>
      <td>Professor Stephen Hawking's funeral will take ...</td>
      <td>2</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Tue Mar 20 17:03:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Un-bee-lievable! 🐝😍 Meet the woman who kept a ...</td>
      <td>3</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Tue Mar 20 16:05:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Arise, Sir Ringo! 🎖🎶@thebeatles drummer @ringo...</td>
      <td>4</td>
      <td>@BBC</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiments_news.to_csv("News_Sentiment_Twitter.csv", index = False)
Date = dt.now().strftime("(%m/%d/%Y)")
```


```python
for each in target_user:
    users_df = sentiments_news.loc[sentiments_news["User"] == each]
    plt.scatter(users_df["Tweets Ago"],users_df["Compound"],label = each)
```


```python
plt.xlim(100,-1)
plt.legend(bbox_to_anchor=(1,1))
plt.title("Sentiment Analysis of News Tweets "+str(Date))
plt.xlabel("Number of Tweets Ago")
plt.ylabel("Tweet sentiment polarity")
plt.savefig("Sentiment Analysis of News Tweets")
plt.grid(True)
plt.show()
```


![png](output_8_0.png)



```python
avg_sent = sentiments_news.groupby("User")
means_sentiments = avg_sent["Compound"].mean()
means_sentiments.head()
```




    User
    @BBC        0.125906
    @CBS        0.299721
    @CNN       -0.068940
    @FoxNews   -0.100056
    @NYTimes   -0.038018
    Name: Compound, dtype: float64




```python
fig, ax = plt.subplots()

x_axis = np.arange(len(means_sentiments))
count2 = 0
count = 0
for sent in means_sentiments:
    ax.text(count2, sent+.01, str(round(sent,2)))
    count2+=1
plt.ylim(-.15,.35)
plt.bar(x_axis, means_sentiments, tick_label = target_user, color = ['r', 'y', 'b', 'g', 'c'])
plt.title("Overall Sentiment of Media Tweets " +str(Date))
plt.xlabel("News Accounts")
plt.ylabel("Tweet Sentiment Polarity")
plt.savefig("Overall Sentiment of News Tweets")
plt.show()
```


![png](output_10_0.png)

