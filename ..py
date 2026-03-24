import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.DataFrame({
    '기생충':   [5, 4, np.nan, 1, 2],
    '어벤져스': [3, np.nan, 4, 5, 4],
    '라라랜드': [4, 5, 1, np.nan, 1],
    '인터스텔라': [np.nan, 3, 5, 4, 5],
    '올드보이': [5, 4, 2, np.nan, 1],
    '스파이더맨': [2, np.nan, 5, 4, 4]
}, index=['유저A', '유저B', '유저C', '유저D', '유저E'])
ratings_filled = ratings.apply(lambda x:x.fillna(x.mean()),axis=1)

sim = cosine_similarity(ratings_filled)
user_sim_df = pd.DataFrame(sim,index=ratings.index,columns=ratings.index)
user_sim_df.round(2)
