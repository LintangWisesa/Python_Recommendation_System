# 1. Popularity based

import numpy as np
import pandas as pd

produk = [
    {'nama': 'Tamiya', 'rating': 5},
    {'nama': 'Drone', 'rating': 4},
    {'nama': 'RC Car', 'rating': 5},
    {'nama': 'Action Figure', 'rating': 3},
    {'nama': 'Bantal', 'rating': 2},
    {'nama': 'Guling', 'rating': 5},
]

dfProduk = pd.DataFrame(produk)
print(dfProduk)

rekomendasiProduk = dfProduk[dfProduk['rating'] == dfProduk['rating'].max()]
print(rekomendasiProduk)