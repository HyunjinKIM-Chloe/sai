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

    s_path = "../../tracks/1213_trial6/"
    test_df = pd.read_csv("../files/freq_test_656rows.csv")  # test용 데이터 (90% * 20%) 확보
    val_df = pd.read_csv("../files/freq_validation_365rows.csv")  # validation용 데이터 (10%) 확보
    test_df.drop(["Unnamed: 0", "cate1"], axis=1, inplace=True)
    val_df.drop(["Unnamed: 0", "cate1"], axis=1, inplace=True)
    info_df = pd.concat([test_df, val_df])

    o_path = list(set(glob.glob(f'{s_path}*/*')))
    path_ls = [file for file in o_path if file.endswith(".mp3") or file.endswith(".wav")]
    name_ls = [name.split('/')[-2] for name in path_ls]
    start_ls = [int(name.split('/')[-1][:-4].split('_')[0]) for name in path_ls]
    end_ls = [int(name.split('/')[-1][:-4].split('_')[1]) for name in path_ls]
    label_ls = [info_df[info_df['name'] == name]['cate2'].iloc[0] for name in name_ls]
    info_df = pd.DataFrame()
    info_df['path'] = path_ls
    info_df['name'] = name_ls
    info_df['start'] = start_ls
    info_df['end'] = end_ls
    info_df['label'] = label_ls
    info_df = info_df.sort_values(by=['name', 'end'])
    info_df.reset_index(drop=True, inplace=True)

    print(len(info_df))
    print(info_df.head)
    result_df = freq.make_frequency_df(tracks_df=info_df)
    print(result_df)
    result_df.to_csv("../files/freq_splited_1213_2.csv", encoding='utf-8-sig')
