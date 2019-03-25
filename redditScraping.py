import praw
import pandas as pd


reddit = praw.Reddit(client_id='RlDJlJX1xEwdTw', client_secret='LbzbnCju7DqJFABepJTblMsCllA', user_agent='nlpSubredditInformation')
# array holds all the topic posts
posts = []

# # get 10 hot posts from the MachineLearning subreddit
# hot_posts = reddit.subreddit('MachineLearning').hot(limit=1000)
# for post in hot_posts:
#     print(post.title)

# # get hottest posts from all subreddits
# hot_posts = reddit.subreddit('all').hot(limit=10)
# for post in hot_posts:
#     print(post.title)


ml_subreddit = reddit.subreddit('MachineLearning')
for post in ml_subreddit.hot(limit=1000):
    posts.append([post.subreddit, post.selftext])


machineLearningPosts = pd.DataFrame(posts, columns=['subreddit', 'body'])
#print(machineLearningPosts.to_string())
export_csv = machineLearningPosts.to_csv(r'/home/dev/Documents/NLP/redditAuthor/dataCSVs/machineLearningPosts.csv', index=None, header=True)
