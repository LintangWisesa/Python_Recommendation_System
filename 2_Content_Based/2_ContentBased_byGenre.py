# kaggle datasets download -d 
# rush4ratio/video-game-sales-with-ratings 
# --unzip

import numpy as np
import pandas as pd

dfGame = pd.read_csv(
    'Video_Games_Sales_as_at_22_Dec_2016.csv'
)
# print(dfGame.head(2))
# print(dfGame.columns.values)
# print(dfGame.isnull().sum())
# print(dfGame[dfGame['Genre'].isnull()])
# print(dfGame.iloc[14246])

dfGame = dfGame.dropna(subset = ['Genre'])
# print(dfGame.isnull().sum())

# =============================================
# content based filtering, feature: 'Genre'
# dataframe: 'Name', 'Platform', 'Genre'
dfGame = dfGame[[ 'Name', 'Platform', 'Genre' ]]

# =============================================
# count genre
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    # ngram_range=(1,2),   # min 2 kata, max 2 kata
    tokenizer = lambda i: i.split('😎'),
    analyzer = 'word',
)

matrixGenre = model.fit_transform(dfGame['Genre'])
Genre = model.get_feature_names()
jumlahGenre = len(Genre)
eventGenre = matrixGenre.toarray()

print(Genre)
print(jumlahGenre)
# print(eventGenre)

# =============================================
# cosinus similarity
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixGenre)
# print(score)

# =============================================
# test model
sayaSuka = 'Duck Hunt'
# print(dfGame[dfGame['Name'] == 'Crash Bandicoot'])
indexSuka = dfGame[dfGame['Name'] == sayaSuka].index.values[0]
# print(indexSuka)

# list all games + cos similarity score
allGames = list(enumerate(score[indexSuka]))
# print(allGames)
gameSama = sorted(
    allGames,
    key = lambda i: i[1],
    reverse = True
)

# =============================================
# show 5 first data, sorted by index
# print(gameSama[:5])
# for i in gameSama[:5]:
#     print(dfGame.iloc[i[0]]['Name'])

# =============================================
# show 5 data randomly, cos sim score > 50%
gameSama50up = []
for i in gameSama:
    if i[1] > 0.5:
        gameSama50up.append(i)
# print(gameSama50up)

import random
rekomendasi = random.choices(gameSama50up, k=5)
print(rekomendasi)
for i in rekomendasi:
    print(
        dfGame.iloc[i[0]]['Name'],
        dfGame.iloc[i[0]]['Platform'],
        dfGame.iloc[i[0]]['Genre']
    )