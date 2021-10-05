import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import os
import glob

import freq_dataset as fd
dataset = fd.MakeFreqDataset()


class CosineSimilarity:
    def __init__(self):
        freq_df = dataset.make_dataframe()
        self.freq_df = freq_df.set_index('name')

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

    def sim_top5(self, cos_df, name):
        chart = cos_df[[name]].sort_values(by=name, ascending=False)
        chart = chart.drop(name) # 자기자신은 삭제

        print(f"\n--*--*--*--*--*--\n{name}와 유사한 곡")
        print("--*--*--*--*--*--\n")
        print(chart.head(5))
        return chart.head(5).index.tolist(), chart.reset_index()