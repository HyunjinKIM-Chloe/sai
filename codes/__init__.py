import warnings
import pandas as pd
import os
import glob
import freq_dataset as freq
warnings.filterwarnings(action='ignore')


if __name__ == '__main__':
    freq = freq.MakeFreqDataset()
    # s_path = "../../sounds/"
    # o_path = list(set(glob.glob(f'{s_path}*/*/*')))  # - set(final_df['path']))
    # path_ls = [file for file in o_path if file.endswith(".mp3") or file.endswith(".wav")]
    # name_ls = [name.split('/')[-1][:-4] for name in path_ls]
    # label_ls = [name.split('/')[-3] for name in path_ls]
    # info_df = pd.DataFrame()
    # info_df['path'] = path_ls
    # info_df['label'] = label_ls
    # info_df['name'] = name_ls
    # print(len(info_df))
    # print(info_df.head)
    # result_df = freq.make_frequency_df(tracks_df=info_df)
    # print(result_df)
    # result_df.to_csv("../files/freq_dataset_1113.csv", encoding='utf-8-sig')

    s_path = "../../sounds/1210/"
    o_path = list(set(glob.glob(f'{s_path}*/*')))
    path_ls = [file for file in o_path if file.endswith(".mp3") or file.endswith(".wav")]
    name_ls = [name.split('\\')[-1][:-4] for name in path_ls]
    label_ls = [name.split('\\')[-2] for name in path_ls]
    # start_ls = [int(name.split('/')[-1][:-4].split('_')[0]) for name in path_ls]
    # end_ls = [int(name.split('/')[-1][:-4].split('_')[1]) for name in path_ls]
    info_df = pd.DataFrame()
    info_df['path'] = path_ls
    info_df['name'] = name_ls
    info_df['label'] = label_ls
    # info_df['start'] = start_ls
    # info_df['end'] = end_ls
    info_df = info_df.sort_values(by=['label', 'name'])
    info_df.reset_index(drop=True, inplace=True)

    print(len(info_df))
    print(info_df.head)
    result_df = freq.make_frequency_df(tracks_df=info_df)
    print(result_df)
    result_df.to_csv("../files/freq_1210.csv", encoding='utf-8-sig')
