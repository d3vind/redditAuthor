#this script is set up to do one user at a time although can eaily be altered to take many
import requests
import time
import pandas as pd


def getCommentsFromPushshift(**kwargs):
    r = requests.get("https://api.pushshift.io/reddit/comment/search/", params=kwargs)
    data = r.json()
    return data['data']


def get_comments_from_reddit_api(comment_ids):
    headers = {'User-agent': 'Comment Collector'}
    params = {}
    params['id'] = ','.join(["t1_" + id for id in comment_ids])
    r = requests.get("https://api.reddit.com/api/info", params=params, headers=headers)
    data = r.json()
    return data['data']['children']


userCommentAuthorList = []
userCommentBodyList = []
before = None
author = "memeinvestor_bot"
while True:
    # only allowed to take 100 at a time so we loop using the created_utc as the last known location to start looking again
    comments = getCommentsFromPushshift(author=author, size=100, before=before, sort='desc', sort_type='created_utc')
    if not comments: break

    # Get the comment ids from push shift to get around comment scraping limit of praw
    comment_ids = []
    for comment in comments:
        before = comment['created_utc']
        comment_ids.append(comment['id'])

    # This will then pass the ids collected from Pushshift and query Reddit's API for the most up to date information
    comments = get_comments_from_reddit_api(comment_ids)

    for comment in comments:
        comment = comment['data']

        # add comments to dataFrame
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
s.to_csv(r'/home/dev/Documents/NLP/redditAuthor/dataCSVs/memeinvestor_bot', index=False)
