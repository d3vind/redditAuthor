#!/usr/bin/env python3

import requests
import time
import pandas as pd


def get_comments_from_pushshift(**kwargs):
    r = requests.get("https://api.pushshift.io/reddit/comment/search/", params=kwargs)
    data = r.json()
    return data['data']


def get_comments_from_reddit_api(comment_ids):
    headers = {'User-agent':'Comment Collector'}
    params = {}
    params['id'] = ','.join(["t1_" + id for id in comment_ids])
    r = requests.get("https://api.reddit.com/api/info", params=params,headers=headers)
    data = r.json()
    return data['data']['children']


userCommentAuthorList = []
userCommentBodyList = []
before = None
author = "srbistan"
while True:
    comments = get_comments_from_pushshift(author=author, size=100, before=before, sort='desc', sort_type='created_utc')
    if not comments: break

    # This will get the comment ids from Pushshift in batches of 100 -- Reddit's API only allows 100 at a time
    comment_ids = []
    for comment in comments:
        before = comment['created_utc'] # This will keep track of your position for the next call in the while loop
        comment_ids.append(comment['id'])

    # This will then pass the ids collected from Pushshift and query Reddit's API for the most up to date information
    comments = get_comments_from_reddit_api(comment_ids)
    # here we can add the comments to a list

    for comment in comments:
        comment = comment['data']

        # Do stuff with the comments
        userCommentBodyList.append((comment['body']))
        userCommentAuthorList.append((comment['author']))
    # sleep to prevent getting picked on by eddit
    time.sleep(2)
s = pd.DataFrame()
se = pd.Series(userCommentBodyList)
se = pd.Series(userCommentAuthorList)

s['author'] = userCommentAuthorList
s['body'] = userCommentBodyList

# print(s.to_string())
# maxwellhill
s.to_csv(r'/home/dev/Documents/NLP/redditAuthor/dataCSVs/srbistan.csv', index=False)

# I'm not sure how often you can query the Reddit API without oauth but once every two seconds should work fine
