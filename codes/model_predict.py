import freq_dataset as fd
import pandas as pd
import numpy as np
import yaml
import json
import subprocess
from datetime import datetime
import os
import glob
import joblib
import warnings
warnings.simplefilter("ignore")


class Preprocessing(fd.MakeFreqDataset):
    def __init__(self):
        super().__init__()
        with open('../config/config.yaml') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.local_path = conf['local_path']
        self.video_path = conf['video_path']
        self.wav_from_video_path = conf['wav_from_video_path']
        self.model_path = conf['model_path']
        self.json_path = conf['json_path']
        self.label_ls = conf['label_ls']
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
                command = f"ffmpeg -i {mp4_name} -ab 160k -ac 2 -ar 44100 -vn {self.wav_from_video_path}{filename}.wav -y"
                subprocess.call(command, shell=True)
        else:
            pass

    def make_info_df(self):
        """
        :return: local_path에 있는 음원을 데이터 프레임으로 정리
                 columns=[path, name]
        """
        o_path = list(set(glob.glob(f'{self.local_path}*'))) + list(set(glob.glob(f'{self.local_path}*/*')))
        path_ls = [file for file in o_path if file.endswith(".mp3") or file.endswith(".wav")]
        name_ls = [name.split('/')[-1][:-4] for name in path_ls]

        info_df = pd.DataFrame()
        info_df['path'] = path_ls
        info_df['name'] = name_ls
        info_df = info_df.sort_values(by=['name'])
        info_df.reset_index(drop=True, inplace=True)
        return info_df

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
        tmp_df = tmp_df.drop(['path', 'duration'], axis=1)
        print(f"이 소리는 {(max(md.predict_proba(tmp_df)[0]) * 100).round(3)}의 확률로 '{md.predict(tmp_df)[0]}' 로 추정됩니다.")
        return md.predict_proba(tmp_df)[0], md.predict(tmp_df)[0]

    def make_predict_df(self, filename=None, start=None, end=None):
        if filename:
            val_df = self.make_frequency_df(tracks_df=self.info_df[self.info_df['name'] == filename])
        else:
            val_df = self.make_frequency_df(tracks_df=self.info_df[start:end])

        final_result = pd.DataFrame()
        for idx in range(len(val_df)):
            freq_df = val_df.iloc[[idx]]
            proba_ls = []
            pred_label = []
            for model in self.model_ls:
                proba, label = self.model_predict(model, freq_df)
                proba_ls.append(proba)
                pred_label.append(label)
            print("----------------")
            result_df = pd.DataFrame(data=proba_ls, columns=self.label_ls).round(5)
            result_df['model'] = [m.split("/")[-1][:-4] for m in self.model_ls]
            result_df['name'] = freq_df['name'][idx]
            result_df['y_pred'] = pred_label
            result_df.sort_values(by=f'{self.label_ls[sum(proba_ls).tolist().index(max(sum(proba_ls)))]}',
                                  ascending=False, inplace=True)
            final_result = pd.concat([final_result, result_df])

        return final_result

    def save_json(self, target_df):
        if len(target_df) == 1:
            new = []
            for idx in range(len(self.label_ls)):
                new.append([f"{self.label_ls[idx]}",
                            f"{round(target_df.iloc[0][idx]*100, 2)}"])
            with open(f"{self.json_path}{target_df.index[0]}.json", "w") as json_file:
                json.dump(new, json_file, indent=True)

        elif len(target_df) > 1:
            total = []  #쌓기코드
            for song in target_df.index.tolist():
                new = []
                row = target_df.loc[song]
                for idx in range(len(self.label_ls)):
                    new.append([f"{self.label_ls[idx]}",
                                f"{round(row[idx]*100, 2)}"])
                with open(f"{self.json_path}{song}.json", "w") as json_file:
                    json.dump(new, json_file, indent=True)
                total.append(new)  #쌓기코드
            with open(f"{self.json_path}{len(target_df)}sounds_{datetime.today().strftime('%Y%m%d')}.json", "w") as json_file:
                json.dump(total, json_file, indent=True)
        else:
            print("Failed to save json file!")

    def predict_check_result(self, filename=None, start=None, end=None):
        result_df = self.make_predict_df(filename, start, end)
        count_df = result_df.groupby('name').sum() / len(self.model_ls)
        for name, item in count_df.iterrows():
            print(f"'{name}' 은 {round(max(item), 2) * 100}% 의 확률로"
                  f" {item.index[item.tolist().index(max(item))]}로 추정됩니다.")
        self.save_json(count_df)
        return count_df
