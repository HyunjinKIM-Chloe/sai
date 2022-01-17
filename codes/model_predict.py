import freq_dataset as fd
import pandas as pd
import numpy as np
import os
import glob
import joblib
import json
import subprocess
from pydub import AudioSegment
import yaml
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.simplefilter("ignore")


class Preprocessing(fd.MakeFreqDataset):
    def __init__(self):
        super().__init__()
        with open('../config/config.yaml') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.video_path = conf['video_path']
        self.wav_from_video_path = conf['wav_from_video_path']
        self.save_path = conf['save_path']
        self.file_path = conf['file_path']
        self.model_path = conf['model_path']
        self.json_path = conf['json_path']
        self.label_ls = ["human", "human_voice", "life", "nature", "song"]
        # local path 에 ffmpeg 가 인식할 수 없는 문자는 치환
        [os.rename(f, f.replace(' ', '')) for f in glob.glob(f"{self.video_path}*")]
        [os.rename(f, f.replace('(', '_')) for f in glob.glob(f"{self.video_path}*")]
        [os.rename(f, f.replace(')', '')) for f in glob.glob(f"{self.video_path}*")]
        [os.rename(f, f.replace(r'\W', '_')) for f in glob.glob(f"{self.video_path}*")]

    def mp4_to_wav(self):
        mp4_ls = [file for file in glob.glob(f'{self.video_path}*') if file.endswith(".mp4")]
        mp4_name_ls = [os.path.basename(p).split('.')[0] for p in mp4_ls].sort()
        wav_name_ls = [os.path.basename(f).split('.')[0] for f in glob.glob(f'{self.wav_from_video_path}*') if f.endswith(".wav")].sort()

        if mp4_name_ls != wav_name_ls:
            for mp4_name in mp4_ls:
                filename = os.path.basename(mp4_name).split('.')[0]
                command = f"ffmpeg -i {mp4_name} -f wav -acodec libmp3lame -ar 44100 -vn {self.wav_from_video_path}{filename}.wav -y"
                subprocess.call(command, shell=True)
        else:
            pass

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

    def edit_frequency_df(self, filename=None, start=None, end=None):
        info_df = self.make_info_df()
        if filename:
            info_df = info_df[info_df['name'] == filename].copy()
        else:
            pass
        freq_df = self.make_frequency_df(info_df, start, end)
        freq_df['start'] = [int(name.split('/')[-1][:-4].split('_')[0]) for name in freq_df['path']]
        freq_df['end'] = [int(name.split('/')[-1][:-4].split('_')[1]) for name in freq_df['path']]
        # freq_df['label'] = [info_df[info_df['name'] == name]['cate2'].iloc[0] for name in freq_df['name']]
        return freq_df

    def load_model_ls(self, filters=None):
        """
        :param filters: 불러올 model의 필터 조건을 regex문법의 string으로 여러 개의 조건을 list type 으로 input
                        ex: ['1213_*XGB*' , '1213_*RF*']
        :return: model list
        """
        if filters is None:
            model_ls = list(set(glob.glob(f'{self.model_path}*')))
        else:
            model_ls = []
            for filter in filters:
                tmp_ls = list(set(glob.glob(f'{self.model_path}{filter}')))
                model_ls += tmp_ls
        print(f"총 {len(model_ls)} 개의 모델이 있습니다.")
        print(model_ls)
        return model_ls


class ModelPredict(Preprocessing):

    def __init__(self):
        super().__init__()
        self.mp4_to_wav()  # mp4 동영상 에서 wav 음원 추출
        self.info_df = self.make_info_df()
        self.model_ls = self.load_model_ls()

    def model_predict(self, model, target_df):
        md = joblib.load(model)
        tmp_df = target_df.set_index('name', drop=True)
        tmp_df = tmp_df.drop(['path', 'duration', 'start', 'end'], axis=1)
        print(f"이 소리는 {(max(md.predict_proba(tmp_df)[0]) * 100).round(3)}의 확률로 '{md.predict(tmp_df)[0]}' 로 추정됩니다.")
        return md.predict_proba(tmp_df)[0], md.predict(tmp_df)[0]

    def make_predict_df(self, filters=None, filename=None, start=None, end=None):
        print(self.label_ls)
        model_ls = self.load_model_ls(filters)  # 조건에 맞는 모델 리스트 가져오기
        val_df = self.edit_frequency_df(filename, start, end)  # save_path에 저장되어있는 음원으로 frequency 생성

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
            result_df['start'] = freq_df['start'][idx]
            result_df['end'] = freq_df['end'][idx]
            result_df['y_pred'] = pred_label
            result_df.sort_values(by=f'{self.label_ls[sum(proba_ls).tolist().index(max(sum(proba_ls)))]}',
                                  ascending=False, inplace=True)
            final_result = pd.concat([final_result, result_df])

        return final_result

    def save_json(self, target_df, name_ls=None):
        if len(target_df) == 1:
            new = []
            for idx in range(len(self.label_ls)):
                new.append([f"{self.label_ls[idx]}",
                            f"{round(target_df.iloc[0][idx]*100, 2)}"])
            with open(f"{self.json_path}{target_df.index[0][0]}_{target_df.index[0][1]}.json", "w") as json_file:
                json.dump(new, json_file, indent=True)

        elif len(target_df) > 1:
            total = []  #쌓기코드
            for idx, song in enumerate(target_df.index.tolist()):
                new = []
                row = target_df.loc[song]
                for idx in range(len(self.label_ls)):
                    new.append([f"{self.label_ls[idx]}",
                                f"{round(row[idx]*100, 2)}"])
                with open(f"{self.json_path}{target_df.index[idx][0]}_{target_df.index[idx][1]}.json", "w") as json_file:
                    json.dump(new, json_file, indent=True)
                total.append(new)  #쌓기코드
            with open(f"{self.json_path}{target_df.index[0][0]}_{target_df.index[0][1]}_{target_df.index[-1][1]}_"
                      f"{datetime.today().strftime('%Y%m%d')}.json", "w") as json_file:
                json.dump(total, json_file, indent=True)
        else:
            print("Failed to save json file!")

    def predict_check_result(self, filters=None, filename=None, start=None, end=None):
        result_df = self.make_predict_df(filters, filename, start, end)
        result_df = result_df.drop(['end'], axis=1)
        result_df.set_index('start', drop=True, inplace=True)
        pct_df = result_df.groupby(['name', 'start']).sum() / len(self.model_ls)
        for name, item in pct_df.iterrows():
            print(f"'{name}' 은 {round(max(item), 2) * 100}% 의 확률로"
                  f" {item.index[item.tolist().index(max(item))]}로 추정됩니다.")
        name_ls = []
        self.save_json(pct_df, name_ls)
        human_pct = (pct_df * 100).iloc[:, :2].sum(axis=1)
        non_human_pct = (pct_df * 100).iloc[:, 2:].sum(axis=1)
        pct = pd.DataFrame(human_pct, columns=['Human'])
        pct['Non-human'] = non_human_pct
        return result_df, pct


class FreqEDA(Preprocessing):
    def __init__(self):
        super().__init__()

    def read_csv(self, filename):
        csvfile = pd.read_csv(f"{self.file_path}{filename}.csv", encoding='utf-8-sig')
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
