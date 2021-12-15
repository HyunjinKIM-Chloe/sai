import freq_dataset as fd
import pandas as pd
import numpy as np
import random
import os
import glob
import joblib
from pydub import AudioSegment
from tqdm import tqdm
import IPython.display
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


class Preprocessing(fd.MakeFreqDataset):
    def __init__(self):
        super().__init__()
        self.save_path = "../../tracks/1213_trial6/"
        self.model_path = "models/"
        self.filepath = "../files/"
        self.label_ls = ['bird', 'cat', 'clap', 'dog', 'drum', 'horse', 'lion', 'shout']

    def cut_song_by_sec(self, song_path, term=3):
        """
        1초 마다 밀면서 3초 간격으로 음원 잘라서 새로운 경로에 저장하는 함수
        :param song_path: 노래 path 입력
        :param term: n초 간격 으로 자르기
        """
        song_name = song_path.split('/')[-1][:-4]
        new_path = f"{self.save_path}{song_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        o_audio = AudioSegment.from_file(song_path)
        for n in range(round(o_audio.duration_seconds)-term+1):
            audio = o_audio[n*1000:(n+term)*1000]  # Works in milliseconds(*1000)
            new_dir = f"{new_path}/{n}_{n+term}.wav"
            audio.export(new_dir, format="wav")  # Export to a wav file in the current path

    def make_info_df(self):
        """
        :return: cut_song_by_sec() 으로 쪼갠 데이터셋 데이터 프레임으로 정리
                 columns=[path, name,시작 초, 끝 초]
        """
        o_path = list(set(glob.glob(f'{self.save_path}*/*')))
        path_ls = [file for file in o_path if file.endswith(".mp3") or file.endswith(".wav")]
        name_ls = [name.split('/')[-2] for name in path_ls]
        start_ls = [int(name.split('/')[-1][:-4].split('_')[0]) for name in path_ls]
        end_ls = [int(name.split('/')[-1][:-4].split('_')[1]) for name in path_ls]

        info_df = pd.DataFrame()
        info_df['path'] = path_ls
        info_df['name'] = name_ls
        info_df['start'] = start_ls
        info_df['end'] = end_ls
        info_df = info_df.sort_values(by=['name', 'end'])
        info_df.reset_index(drop=True, inplace=True)
        return info_df

    def edit_frequency_df(self, start=None, end=None):
        info_df = self.make_info_df()
        freq_df = self.make_frequency_df(info_df, start, end)
        freq_df['start'] = [int(name.split('/')[-1][:-4].split('_')[0]) for name in freq_df['path']]
        freq_df['end'] = [int(name.split('/')[-1][:-4].split('_')[1]) for name in freq_df['path']]
        # freq_df['label'] = [info_df[info_df['name'] == name]['cate2'].iloc[0] for name in freq_df['name']]
        return freq_df


class ModelPredict(Preprocessing):
    def __init__(self):
        super().__init__()

    def load_model_ls(self, filters):
        """
        :param filters: 불러올 model의 필터 조건을 regex문법의 string으로 여러 개의 조건을 list type 으로 input
                        ex: ['1213_*XGB*' , '1213_*RF*']
        :return: model list
        """
        model_ls = []
        for filter in filters:
            tmp_ls = list(set(glob.glob(f'{self.model_path}{filter}')))
            model_ls += tmp_ls
        print(f"총 {len(model_ls)} 개의 모델이 있습니다.")
        print(model_ls)
        return model_ls

    def model_predict(self, model, target_df):
        md = joblib.load(model)
        tmp_df = target_df.set_index('name', drop=True)
        tmp_df = tmp_df.drop(['path', 'duration', 'start', 'end'], axis=1)
        print(f"이 소리는 {(max(md.predict_proba(tmp_df)[0]) * 100).round(3)}의 확률로 '{md.predict(tmp_df)[0]}' 로 추정됩니다.")
        return md.predict_proba(tmp_df)[0], md.predict(tmp_df)[0]

    def make_predict_df(self, filters, start=None, end=None):
        print(self.label_ls)
        model_ls = self.load_model_ls(filters)  # 조건에 맞는 모델 리스트 가져오기
        val_df = self.edit_frequency_df(start, end)  # save_path에 저장되어있는 음원으로 frequency 생성

        final_result = pd.DataFrame()
        for idx in tqdm(range(len(val_df))):
            freq_df = val_df.iloc[[idx]]
            proba_ls = []
            pred_label = []
            for model in model_ls:
                proba, label = self.model_predict(model, freq_df)
                proba_ls.append(proba)
                pred_label.append(label)
            print("----------------")
            result_df = pd.DataFrame(data=proba_ls, columns=self.label_ls).round(1)
            result_df['model'] = [m.split("/")[-1][:-4] for m in model_ls]
            result_df['name'] = freq_df['name'][idx]
            result_df['y_pred'] = pred_label
            result_df.sort_values(by=f'{self.label_ls[sum(proba_ls).tolist().index(max(sum(proba_ls)))]}',
                                  ascending=False, inplace=True)
            final_result = pd.concat([final_result, result_df])

        return final_result


class FreqEDA(Preprocessing):
    def __init__(self):
        super().__init__()

    def read_csv(self, filename):
        csvfile = pd.read_csv(f"{self.filepath}{filename}.csv", encoding='utf-8-sig')
        try:
            csvfile.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            pass
        return csvfile

    def count_label(self, df, colname='cate2'):
        if type(df) == str:  # csv 파일을 불러 오고 싶은 경우 string 으로 filename 입력
            target_df = self.read_csv(df)
        else:
            target_df = df # DataFrame 으로 바로 확인 하고 싶은 경우에 사용

        print(target_df.groupby(colname).count()['name'])
        count_df = target_df.groupby(colname).count().sort_values(by='name', ascending=False)
        plt.figure(figsize=(12, 8))
        return plt.bar(count_df.index, height=count_df['name'])

    def del_outlier(self, df, max=20, min=1):
        """
        :param df: 데이터를 csv 또는 DataFrame 형식으로 입력
        :param max: max 초 이상인 아웃라이어 음원 제거
        :param min: min 초 이하인 아웃라이어 음원 제거
        :return: 아웃라이어가 제거된 최종 DataFrame return하며 label 개수와 duration Boxplot 시각화
        """
        if type(df) == str:  # csv 파일을 불러 오고 싶은 경우 string 으로 filename 입력
            target_df = self.read_csv(df)
        else:
            target_df = df # DataFrame 으로 바로 확인 하고 싶은 경우에 사용

        long_out = target_df[target_df['duration'] > max].index.tolist()
        short_out = target_df[target_df['duration'] < min].index.tolist()
        print("너무 긴 outlier 제외: ", len(long_out))
        print("너무 짧은 outlier 제외: ", len(short_out))
        final_df = target_df.drop(long_out + short_out, axis=0)
        print(self.count_label(final_df))
        print(sns.boxplot(data=final_df, x='cate2', y='duration'))

        return final_df
