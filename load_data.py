import pandas as pd
from surprise import Dataset
from surprise import Reader

# loading our .csv and filtering them for use later
music = pd.read_csv(r"music.csv")
ratings = pd.read_csv(r"ratings.csv")

filterSong_ID = music["song_id"]
filterUser_ID = ratings["user_id"]
filterRating = ratings["rating"]

# creating a dict to load or separate data sets into a single dataFrame
train_dict = {
    "item": filterSong_ID,
    "user": filterUser_ID,
    "rating": filterRating
}

# dataFrame
train_df = pd.DataFrame(train_dict)

# reader is used to parse through a dataset that contains ratings.
reader = Reader(rating_scale=(1, 10))

# Loads pandas data frame
data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)



###########################################################################################
# OLD TEST CODE DOWN BELOW ( for reference )
###########################################################################################

# This is the same data that was plotted for similarity earlier
# with one new user "E" who has rated only movie 1

# ratings_dict = {
#     "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
#     "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
#     "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
# }
# ratings_df = pd.DataFrame(ratings_dict)
# reader = Reader(rating_scale=(1, 5))
#
# # Loads Pandas dataframe
# data = Dataset.load_from_df(ratings_df[["user", "item", "rating"]], reader)
