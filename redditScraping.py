import praw


reddit = praw.Reddit(client_id='RlDJlJX1xEwdTw', client_secret='LbzbnCju7DqJFABepJTblMsCllA', user_agent='nlpSubredditInformation')


# get 10 hot posts from the MachineLearning subreddit
hot_posts = reddit.subreddit('MachineLearning').hot(limit=10)
for post in hot_posts:
    print(post.title)

# get hottest posts from all subreddits
hot_posts = reddit.subreddit('all').hot(limit=10)
for post in hot_posts:
    print(post.title)
