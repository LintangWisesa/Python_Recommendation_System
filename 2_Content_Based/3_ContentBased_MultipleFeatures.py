import numpy as np
import pandas as pd

dfGame = pd.read_csv(
    'Video_Games_Sales_as_at_22_Dec_2016.csv'
)
dfGame = dfGame.dropna(subset = ['Genre', 'Platform'])
dfGame = dfGame[[ 'Name', 'Platform', 'Genre' ]]
# print(dfGame.head(10))

# ===========================================
# add a new col: 'Platform' + 'Genre'
def mergeCol(i):
    return str(i['Platform']) + ' ' + str(i['Genre'])

dfGame['features'] = dfGame.apply(mergeCol, axis = 1)
# print(dfGame.head(10))
# print(len(dfGame['Platform'].unique()))
# print(len(dfGame['Genre'].unique()))

# ===========================================
# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(tokenizer=lambda x: x.split(' '))
matrixFeature = model.fit_transform(dfGame['features'])

features = model.get_feature_names()
jmlFeatures = len(features)
# print(features)
# print(jmlFeatures)
# print(matrixFeature.toarray()[0])

# ===========================================
# cosinus similarity
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixFeature)
# print(score[0])

# ===========================================
# testing
sukaGame = 'Tetris'
indexSuka = dfGame[dfGame['Name'] == sukaGame].index.values[0]
print(indexSuka)

daftarScore = list(enumerate(score[indexSuka]))
# print(daftarScore)

sortDaftarScore = sorted(
    daftarScore,
    key = lambda j: j[1],
    reverse = True
)
# print(sortDaftarScore[:5])

# ======================================
# show top 5 similar recommended games randomly

similarGames = []
for i in sortDaftarScore:
    if i[1] > .8:
        similarGames.append(i)
# print(similarGames)

import random
rekomendasi = random.choices(similarGames, k=5)
# print(rekomendasi)

for i in rekomendasi:
    data = dfGame.iloc[i[0]].values
    print(data[0], data[1], data[2])
