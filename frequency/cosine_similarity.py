import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import access_db as db


class CosineSimilarity(db.Query):
    def __init__(self):
        super().__init__()
        freq_df = self.read_db('frequency')
        if len(freq_df) == 0:
            pass
        else:
            freq_df.drop('id', axis=1, inplace=True)
            self.freq_df = freq_df.set_index('track_id')

    def make_cos_df(self, z_scale=True):
        if z_scale:
            input_df = (self.freq_df - np.mean(self.freq_df)) / np.std(self.freq_df)
        else:
            input_df = self.freq_df

        cos = cosine_similarity(input_df)
        print("shape: ", cos.shape)
        cos_df = pd.DataFrame(cos)
        cos_df.set_index(self.freq_df.index, inplace=True)
        cos_df.columns = self.freq_df.index
        return cos_df

    def make_similarity_df(self):
        cos_df = self.make_cos_df()
        result_df = pd.DataFrame()
        s1, s2, cos = [], [], []
        for idx, x in enumerate(range(len(cos_df.index))):
            for y in range(idx, len(cos_df.index)):
                s1.append(cos_df.columns[x])
                s2.append(cos_df.columns[y])
                cos.append(cos_df.iloc[x, y])
        result_df['track1'] = s1
        result_df['track2'] = s2
        result_df['similarity'] = cos
        result_df = result_df.drop(result_df[result_df['track1'] == result_df['track2']].index)
        return result_df
