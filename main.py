from load_data import data
import pandas as pd
from scipy import spatial
from surprise import KNNWithMeans
import numpy as np  # not being used... yet

# used for item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False,  # computes similarities between items
}
# stuffing our algorithm into a variable for the main file
algorithm = KNNWithMeans(sim_options=sim_options)

# putting our training data into the similarity algorithm
trainingSet = data.build_full_trainset()
algorithm.fit(trainingSet)

# making a dataFrame with our comparison data
ratings = pd.read_csv(r"ratings.csv")
ratings_df = pd.DataFrame(ratings, columns=['user_id', 'rating'])
user_id = ratings_df['user_id']
rat = ratings_df['rating']

music = pd.read_csv(r"music.csv")
music_df = pd.DataFrame(music)
genre = music_df['Genre']

filteredRatings = {  # song_id, rating in that order
    "song_id": [],
    "rating": []
}

# creating an array for the similarity comparison data to be stuffed into
user_0 = []
user_1 = []
user_2 = []
user_3 = []
user_4 = []
user_5 = []

# creating our user's survey into an array
user_array = []

print("\nWelcome to the Music Recommenderer 3000")
print("\nOur program will measure how compatible your music tastes are with anonymous users."
      "\nThen it will find a song from our list that best suits your style!")
print("\nBut first we are going to ask you to rate different genres from 1-10 (1 being never listen to, and 10 being "
      "absolutely in love with)\n")

user_metal = int(input("Metal: "))
user_country = int(input("Country: "))
user_pop = int(input("Pop: "))
user_punk = int(input("Punk: "))

for i in range(97):
    if genre[i] == 'Metal':
        user_array.append(user_metal)
    if genre[i] == 'Country':
        user_array.append(user_country)
    if genre[i] == 'Pop':
        user_array.append(user_pop)
    if genre[i] == 'Punk':
        user_array.append(user_punk)

# for loop to find individual users within our comparison data to single out and put into separate arrays.
for i in range(len(ratings_df['rating'])):
    if user_id[i] == 0:
        user_0.append(rat[i])
    if user_id[i] == 1:
        user_1.append(rat[i])
    if user_id[i] == 2:
        user_2.append(rat[i])
    if user_id[i] == 3:
        user_3.append(rat[i])
    if user_id[i] == 4:
        user_4.append(rat[i])
    if user_id[i] == 5:
        user_5.append(rat[i])

# making an array to store the values from our for loop above for easier comparison and reference
spatial_result = [spatial.distance.cosine(user_array, user_0), spatial.distance.cosine(user_array, user_1),
                  spatial.distance.cosine(user_array, user_2), spatial.distance.cosine(user_array, user_3),
                  spatial.distance.cosine(user_array, user_4), spatial.distance.cosine(user_array, user_5)]

lowest_spatial_match = min(spatial_result)
index_num = spatial_result.index(lowest_spatial_match)
max_spatial_rating = str("user_" + str(index_num))

for key in range(len(ratings)):  # making a for loop that runs through the ratings dataset and pulls out the wanted song_id
    # TODO: replace the 9 with the ai's output
    if ratings.user_id[key] == index_num:  # checking if the song_id is what we want

        tempList = []  # making a temp list
        tempSongID = ratings.song_id[key]  # pulling the song_id from the dataset
        tempRatingID = ratings.rating[key]  # pulling the rating from the dataset

        tempList.append(tempSongID)  # adding songId to the templist
        tempList.append(tempRatingID)  # adding the rating to the temp list

        filteredRatings['song_id'].append(tempSongID)  # adding the songID to the dict
        filteredRatings['rating'].append(tempRatingID)  # adding the rating to the dict

pullRating = filteredRatings["rating"]  # pulling the ratings column out of filteredRatings
pullSongID = filteredRatings["song_id"]  # pulling the song_id column out of pullRating

max_rating = max(pullRating)                 # finding the max value in pullRating
indexOfMax = pullRating.index(max_rating)    # finding the index of the max value inside
                                             # pullRating (what spot does max show up inside the list)

songTitle = ""                                              # making a blank value for songTitle
artistName = ""

for key in range(len(music)):                               # making a for loop
    if music.song_id[key] == pullSongID[indexOfMax]:     # checking if the song id is what we are looking for
        artistName = music.Artist[key]
        songTitle = music.SongTitle[key]                    # adding the SongTitle to the songTitle var

# print("songTitle: ", songTitle)
# print("artist: ", artistName)

# print("DATA: ", spatial_result)  # used for testing
# print("LOWEST: ", lowest_spatial_match)  # used for testing
# print("INDEX #: ", index_num)  # used for testing

print("\n0 = Total Match\n1 = Polar Opposite\n")

print("Your match score is: ")
print(lowest_spatial_match)

print("\nSo this would mean you are based matched with User", index_num)

# Now we would take a song that the matched user likes and recommend it to our user.
print("\nUser", index_num, "rated '", songTitle, "' by '", artistName, "' :", max_rating)
print("\nWe think you might like this song too!\n\n")

############################################################################################
# OLD TEST CODE DOWN BELOW ( for reference )
############################################################################################

# user_id0 = [1, 2, 3, 4]
# user_id1 = [2, 4, 2, 4]
# user_id2 = [2.5, 4, 1.5, 1]
# user_id3 = [4.5, 5, 4, 4.5]

# spatial_result = [spatial.distance.cosine(user, user_id0), spatial.distance.cosine(user, user_id1),
#                   spatial.distance.cosine(user, user_id2), spatial.distance.cosine(user, user_id3)]


###############################################################################################

#
# if prediction.est > 3:
#     print("SUBJECT E WOULD LIKE THIS")
# else:
#     print("SUBJECT E WOULD NOT LIKE THIS")


# prediction = algo.predict(1, 20)
# print(prediction.est)

# list = []

# for i in range(1, 6):
#     for x in range(1, 98):
#         prediction = algo.predict(i, x)
#         print("i: ", i, "x: ", x, "prediction: ", prediction.est, "\n")
#         # list.append(prediction.est)
#         # if list.index(round(prediction.est)) == false:
#         #     list.append(prediction.est)


# TODO:
#     1. get non-randomized data for our database
#     2. find the user_id with the closest match to our user_input
#       2.1 create user array filled with zeroes and create a questionnaire that selects random songs from
#           the music master list and asks the user to rate them.
#       2.2 test and make sure it will still give accurate results with blank spaces or zeroes
#     3. take that user_id in and find their highest rating from the algo.prediction
#     4. find the song_id from that and output the song title and artist from our music list.
