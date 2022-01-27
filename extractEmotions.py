import os
import pandas as pd
import re
import time

from pysentimiento import create_analyzer

# the functon for pre-processing: removing noise
def removeNoise(text):

    CLEANR = re.compile('<.*?>')
    text_wo_url = []

    for tweet in text:
        # removes url
        cleaned_url = re.sub(r'http\S+', '', tweet, flags=re.MULTILINE)
        # removes mention --> !!! username is also removed
        cleaned_url_mention = re.sub("@[A-Za-z0-9_]+","", cleaned_url)
        # removes html tag
        clean_url_mention_html = re.sub(CLEANR, '', cleaned_url_mention)
        text_wo_url.append(clean_url_mention_html)

    return text_wo_url


def createEmotionsFeatures(tweet_texts, analyzer):
    """Expects a multidimensional array of texts and an analyzer object from Pysentimento.
    Makes predictions for the provided text streams and outputs the probabilities of either emotions
    or sentiments embedded in tweets.
    Datatypes:
    > inputs: multidimensional array of strings, Pysentimento analyzer object
    > outputs: a dictionary of emotions/sentiments probability"""
    
    emotions_list = []
    i = 0

    for text in tweet_texts:
        emotions_list.append(analyzer.predict(text).probas) 
        i += 1
        percent_done = round((i / len(tweet_texts)) * 100, 2)
        print(f"{percent_done}% done ...", end='\r')

    return emotions_list

def emotionalDirection(emotions_dataframe, joy_weight, sadness_weight, anger_weight, disgust_weight, surprise_weight=0, fear_weight=0, others_weight=0):
    """Expects a dataframe with emotion names (column names) and probabilities of these emotions in each tweet.
    The probabilities yielded by the model reflect the certainty of presence of these emotions in the Tweets.
    Therefore, if anger is evident, it will have a high probability in the row corresponding to the analyzed tweet.
    These probabilities thus can be treated as weights allowing to measure the intensity of the emotions embedded in
    the tweet.

    
    If the tweet is positive, it will have a positive score, with a maximal value of joy_weight
    (provided that the likelihood of joy is 1.0)
    If it expresses sadness, it will have a slightly negative score, with a maximal value of sadness_weight
    (provided that the likelihood of sadness is 1.0)
    
    Such strong, negative emotions as anger and disgust should make the direction score evidently negative,
    thus their weights are suggested to be negative numbers, lower than the weight for sadness.

    Other categories of emotions such as fear, surprise and neutral mixture of emotions (others) are suggested to
    have weights equal to 0 or a near-zero value in the context of Twitter emotions analysis related to positive/hateful
    tweets.
    """
    
    return emotions_dataframe["joy"] * joy_weight + \
           emotions_dataframe["sadness"] * sadness_weight + \
           emotions_dataframe["anger"] * anger_weight + \
           emotions_dataframe["disgust"] * disgust_weight + \
           emotions_dataframe["surprise"] * surprise_weight + \
           emotions_dataframe["fear"] * fear_weight + \
           emotions_dataframe["others"] * others_weight
    

def main():
    
    # program's driver procedure
    # Script's convention: camel case for functions, snake case for variables
    
    # indication of function's failure
    exitcode = -1
    
    # program's time control
    start = time.perf_counter()
    
    # load the dataset
    twitter_data = pd.read_csv("data/most_liked_tweets.csv")
    if twitter_data.columns[0] == "Unnamed: 0":
        twitter_data.drop(columns=["Unnamed: 0"], inplace=True)
    else:
        pass

    # create a folder for saving the new dataset
    if os.path.isdir("data/"):
        pass
    else:
        os.mkdir("data")

    # new dataset creation: the probabilities of emotions embedded in tweets
    # emotions intensity assigned according the the recognized emotions and to the assigned weights
    try:
        # initialize the variable for Pysentimento object with the analyzer
        analyzer = create_analyzer(task="emotion", lang="en")

        # tweets corpus extraction
        tweets = twitter_data.tweet
        
        # perform text cleaning: removal of URLs, usernames, HTML tags
        cleaned_tweets = removeNoise(tweets)

        # extract the emotions
        print("\n")
        emotions = createEmotionsFeatures(cleaned_tweets, analyzer)
        print("\n")
        
        # create a dataframe object out of the retrieved emotions
        emotions_df = pd.DataFrame(emotions)

        # Applicable just to the emotions, not sentiments!
        # obtain intensities of emotions embedded in the tweets
        emotions_direction_df = pd.DataFrame(emotionalDirection(pd.DataFrame(emotions),
                                                                joy_weight=10,
                                                                sadness_weight=-1,
                                                                anger_weight=-10,
                                                                disgust_weight=-10,
                                                                surprise_weight=0,
                                                                fear_weight=0,
                                                                others_weight=0), columns=["intensity"])
        # create a dataframe object out of the retrieved intensities
        emotions_intensity_df = pd.concat([emotions_df, emotions_direction_df], axis=1)

        # concatenate the initial dataframe and the dataframe for emotions and the measured intensities
        output_df = pd.concat([twitter_data, emotions_intensity_df], axis=1)
        
        # save the output dataframe to CSV file
        output_df.to_csv("data/tweets_emotions_dataset.csv")
        
        exitcode = 0
    
    except Exception as e:
        print("Error occured. %s" % e)
        
    # print the execution total time:
    finish = time.perf_counter()
    print(f"\n[ INFO ] Total execution time: {round(finish - start, 2)} second(s)")
    # if function executed successfully, return exit code of 0
    return exitcode

if __name__ == "__main__":
    main()