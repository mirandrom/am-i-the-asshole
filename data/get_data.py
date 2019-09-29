from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
from collections import defaultdict

submission_filter = [
    'author',
    'author_fullname',
    'full_link',
    'is_self',
    'num_comments',
    'score',
    'selftext',
    'title',
    'id',
]

comment_filter = [
    'author',
    'author_fullname',
    'body',
    'is_submitter',
    'id',
    'link_id', # post id
    'parent_id', # parent id = link id when top level comment
    'score',
    'total_awards_received',
]


def clean_posts(df):
    df = df.loc[(df["title"].str.startswith("AITA")) | (df["title"].str.startswith("WIBTA"))]
    df = df.loc[~(df["selftext"] == "[removed]")]
    df = df.loc[~(pd.isna(df["selftext"]))]
    df = df.loc[df.selftext == ""]
    df = df.loc[df["num_comments"] > 0]
    return df


def clean_comments(df, post_ids):
    df = df.loc[df["parent_id"] == df["link_id"]]
    df["link_id"] = df["link_id"].apply(lambda x: x[3:])
    df = df.loc[df["link_id"].isin(post_ids)]

    def find_labels(text: str):
        return [q for q in ["NTA", "YTA", "ESH", "NAH", "INFO"] if q in text]

    df["labels"] = df["body"].apply(lambda x: find_labels(x))
    df["num_labels"] = df["labels"].apply(lambda x: len(x))
    df = df.loc[df["num_labels"] == 1]
    df["labels"] = df["labels"].apply(lambda x: x[0])
    return df


def merge_comments_and_posts(df_posts: pd.DataFrame, df_comments: pd.DataFrame):
    itol = ["NTA", "YTA", "ESH", "NAH", "INFO"]
    ltoi = {l:i for i,l in enumerate(itol)}
    print("cleaning posts")
    l = len(df_posts)
    df_posts = clean_posts(df_posts)
    post_ids = df_posts.id.to_list()
    print(f"{l - len(df_posts)} posts removed, cleaning comments")
    l = len(df_comments)
    df_comments = clean_comments(df_comments, post_ids)
    print(f"{l - len(df_comments)} comments removed, merging posts and comments")
    comment_labels = df_comments.labels.to_list()
    comment_post_ids = df_comments.link_id.to_list()
    comment_score = df_comments.score.to_list()
    post_labels_dict = {post_id: [0,0,0,0,0] for post_id in post_ids}
    for post_id, label, score in zip(comment_post_ids, comment_labels, comment_score):
        post_labels_dict[post_id][ltoi[label]] += score
    print("updating df_posts with labels")
    df_posts["label_counts"] = [post_labels_dict[post_id] for post_id in post_ids]
    df_posts["label_sum"] = df_posts["label_counts"].apply(lambda x: sum(x))
    l = len(df_posts)
    df_posts = df_posts[df_posts["label_sum"] > 0]
    df_posts["label_probs"] = [[c/s for c in counts] for counts, s in zip(
        df_posts["label_counts"], df_posts["label_sum"])]

    print(f"{l - len(df_posts)} posts removed")
    df_posts.to_pickle("aita_2019_posts_labeled.pkl")
    df_comments.to_pickle("aita_2019_comments_cleaned.pkl")


if __name__ == '__main__':
    # api = PushshiftAPI()
    # start_dt = int(dt.datetime(2019, 1, 1).timestamp())
    # posts_gen = api.search_submissions(
    #     after=start_dt,
    #     subreddit="amitheasshole",
    #     filter=submission_filter
    # )
    # df_posts = pd.DataFrame()
    # posts = []
    # for post in posts_gen:
    #     posts.append(post.d_)
    #     if len(posts) == 25_000:
    #         df_posts = df_posts.append(pd.DataFrame(posts))
    #         print(f"Writing {len(df_posts)} posts to df...")
    #         df_posts.to_pickle("aita_2019_posts.pkl")
    #         posts = []
    #
    # df_comments = pd.DataFrame()
    # for q in ["NTA", "YTA", "ESH", "NAH", "INFO"]:
    #     print("Getting comments for ", q)
    #     comments_gen = api.search_comments(
    #         after=start_dt,
    #         subreddit="amitheasshole",
    #         filter=comment_filter,
    #         q=q,
    #     )
    #     comments = []
    #     for comment in comments_gen:
    #         comments.append(comment.d_)
    #         if len(comments) == 10_000:
    #             print(f"scraped {len(comments)} comments")
    #         if len(comments) == 100_000:
    #             df_comments = df_comments.append(pd.DataFrame(comments))
    #             print(f"Writing {len(df_comments)} posts to df...")
    #             df_comments.to_pickle("aita_2019_comments.pkl")
    #             break
    #     df_comments.to_pickle("aita_2019_comments.pkl")

    df_posts = pd.read_pickle("aita_2019_posts.pkl")
    df_comments = pd.read_pickle("aita_2019_comments.pkl")
    merge_comments_and_posts(df_posts, df_comments)
