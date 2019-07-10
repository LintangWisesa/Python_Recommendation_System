# 2. Content Based Filtering
# cosinus similarity

data = [
    'Buku Komik Sejarah',
    'Buku Komik Politik',
    'Buku Hobi Kuliner',
    'Buku Kuliner Hobi',
    'Sejarah Komik Buku',
]

from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer()
hitung = model.fit_transform(data)

# print(model.get_feature_names())
# print(hitung.toarray())

from sklearn.metrics.pairwise import cosine_similarity
similarityScore = cosine_similarity(hitung)
# print(similarityScore)

sayaSuka = 2
# print(similarityScore[0])
# print(list(enumerate(similarityScore[sayaSuka])))
print(sorted(
    list(enumerate(similarityScore[sayaSuka])),
    key = lambda x: x[1],
    reverse=True
    )
)
