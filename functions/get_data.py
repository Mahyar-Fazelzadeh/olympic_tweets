import snscrape.modules.twitter as sntwitter
import pandas as pd


def scrape_data(query, n_tweets):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        
        if len(tweets) == n_tweets:
            break
        else:
            tweets.append([tweet.content])
    df = pd.DataFrame(tweets, columns=['Tweet'])
    return df