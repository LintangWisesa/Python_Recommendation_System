# Collaborative Filtering

import numpy as np
import pandas as pd

df = pd.read_excel('dataTV.xlsx', index_col=0)
df.fillna(0, inplace=True)
print(df)

# ================================
# correlation

dfCorr = df.corr(method='pearson')
print(dfCorr)
# pearson : standard correlation coefficient
# kendall : Kendall Tau correlation coefficient
# spearman : Spearman rank correlation

# ==============================
# testing

saya = [
    ('sinetron_A', 1), ('kartun_A', 5)
]

skorSama = pd.DataFrame()
for produk, rating in saya:
    skor = dfCorr[produk] * (rating - 2.5)
    skor = skor.sort_values(ascending = False)
    skorSama = skorSama.append(skor, ignore_index=False)

print(skorSama)
print(skorSama.sum().sort_values(ascending=False))