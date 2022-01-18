import freq_dataset as fd
import access_db as db
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
sql = db.Query()
cursor, engine, db = sql.connect_sql()


class Preprocessing(fd.MakeFreqDataset):
    def __init__(self):
        super().__init__()
        with open('../config/config.yaml') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.video_path = conf['video_path']
        self.wav_from_video_path = conf['wav_from_video_path']
        self.split_wav_path = conf['split_wav_path']
        self.file_path = conf['file_path']
        self.model_path = conf['model_path']
        self.IP = conf['address']
        self.label_ls = ["human", "human_voice", "life", "nature", "song"]
        self.videos_df = sql.read_db('videos')

    def mp4_to_wav(self, video_id):
        wav_name_ls = [os.path.basename(f).split('.')[0] for f in glob.glob(f'{self.wav_from_video_path}*') if
                       f.endswith(".wav")]
        try:
            if str(video_id) not in wav_name_ls:
                uri = self.videos_df[self.videos_df['id'] == video_id]['uri'].iloc[0]
                command = f"ffmpeg -i https://{self.IP}/video/{uri} -f wav -acodec libmp3lame -ar 44100 -vn " \
                          f"{self.wav_from_video_path}{video_id}.wav -y"
                subprocess.call(command, shell=True)
            else:
                pass
                print(f"id-{video_id} video has already been converted to wav file.")
        except:
            print("Failed to load video file! please check video_id.")

    def cut_song_by_msec(self, video_id, term=3):
        """
        0.1초 마다 밀면서 3초 간격으로 음원 잘라서 새로운 경로에 저장하는 함수
        :param song_path: 노래 path 입력
        :param term: n초 간격 으로 자르기
        """
        song_path = f"{self.wav_from_video_path}{video_id}.wav"
        new_path = f"{self.split_wav_path}{video_id}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)

            o_audio = AudioSegment.from_file(song_path)
            for ms in range((round(o_audio.duration_seconds)-round(term, 0)+1)*10):
                ms = ms/10
                audio = o_audio[ms*1000:(ms+term)*1000]  # Works in milliseconds(*1000)
                new_dir = f"{new_path}/{ms}_{ms+term}.wav"
                audio.export(new_dir, format="wav")  # Export to a wav file in the current path
        else:
            print(f"{video_id}.wav already has split files")

    def make_info_df(self):
        """
        :return: cut_song_by_msec() 으로 쪼갠 데이터셋 데이터 프레임으로 정리
                 columns=[path, name,시작 초, 끝 초]
        """
        o_path = list(set(glob.glob(f'{self.split_wav_path}*/*')))
        path_ls = [file for file in o_path if file.endswith(".mp3") or file.endswith(".wav")]
        name_ls = [name.split('/')[-2] for name in path_ls]
        start_ls = [float(name.split('/')[-1][:-4].split('_')[0]) for name in path_ls]
        end_ls = [float(name.split('/')[-1][:-4].split('_')[1]) for name in path_ls]

        info_df = pd.DataFrame()
        info_df['path'] = path_ls
        info_df['name'] = name_ls
        info_df['start'] = start_ls
        info_df['end'] = end_ls
        info_df = info_df.sort_values(by=['name', 'end'])
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

    def save_frequency_df(self, video_id=None, start=None, end=None):
        info_df = self.make_info_df()
        if video_id:
            info_df = info_df[info_df['name'] == str(video_id)].copy()
        else:
            pass
        freq_df = self.make_frequency_df(info_df, start, end)
        freq_df['start'] = [float(name.split('/')[-1][:-4].split('_')[0]) for name in freq_df['path']]
        freq_df['end'] = [float(name.split('/')[-1][:-4].split('_')[1]) for name in freq_df['path']]
        freq_df.drop(['path', 'duration'], axis=1, inplace=True)
        freq_df.rename(columns={'name': 'video_id'}, inplace=True)
        return freq_df


class ModelPredict(Preprocessing):

    def __init__(self):
        super().__init__()
        [self.mp4_to_wav(v_id) for v_id in self.videos_df['id'].tolist()]  # mp4 동영상 에서 wav 음원 추출
        [self.cut_song_by_msec(v_id) for v_id in self.videos_df['id'].tolist()]  # wav 음원을 0.1초 단위로 잘라 재저장
        self.info_df = self.make_info_df()
        self.model_ls = self.load_model_ls()

    def model_predict(self, model, target_df):
        md = joblib.load(model)
        tmp_df = target_df.set_index('video_id', drop=True)
        tmp_df = tmp_df.drop(['start', 'end'], axis=1)
        # print(f"이 소리는 {(max(md.predict_proba(tmp_df)[0]) * 100).round(3)}의 확률로"
        #       f"'{md.predict(tmp_df)[0]}' 로 추정됩니다.")
        return md.predict_proba(tmp_df)[0], md.predict(tmp_df)[0]

    def make_predict_df(self, video_id, start=None, end=None):
        print(self.label_ls)
        # sound_features 에 저장되어있는 frequency 가져오기
        query = f"SELECT * FROM sound_features WHERE video_id = {video_id}"
        cursor.execute(query)
        result = cursor.fetchall()
        val_df = pd.DataFrame(result)
        if start:
            val_df = val_df[val_df['start'] >= start][val_df['start'] < end]
        else:
            pass
        print(f"총 {len(val_df)}건의 데이터를 가져왔습니다.")

        final_result = pd.DataFrame()
        for idx in tqdm(range(len(val_df))):
            freq_df = val_df.iloc[[idx]]
            proba_ls = []
            pred_label = []
            for model in self.model_ls:
                proba, label = self.model_predict(model, freq_df)
                proba_ls.append(proba)
                pred_label.append(label)
            result_df = pd.DataFrame(data=proba_ls, columns=self.label_ls).round(1)
            result_df['model'] = [m.split("/")[-1][:-4] for m in self.model_ls]
            result_df['video_id'] = freq_df['video_id'][idx]
            result_df['start'] = freq_df['start'][idx]
            result_df['end'] = freq_df['end'][idx]
            result_df['y_pred'] = pred_label
            result_df.sort_values(by=f'{self.label_ls[sum(proba_ls).tolist().index(max(sum(proba_ls)))]}',
                                  ascending=False, inplace=True)
            final_result = pd.concat([final_result, result_df])
        return final_result

    def predict_check_result(self, video_id, start=None, end=None):
        result_df = self.make_predict_df(video_id, start, end)
        result_df = result_df.drop(['end'], axis=1)
        result_df.set_index('start', drop=True, inplace=True)
        pct_df = result_df.groupby(['video_id', 'start']).sum() / len(self.model_ls)
        # for name, item in pct_df.iterrows():
        #     print(f"'{name}' 은 {round(max(item), 2) * 100}% 의 확률로"
        #           f" {item.index[item.tolist().index(max(item))]}로 추정 됩니다.")
        human_pct = (pct_df * 100).iloc[:, :2].sum(axis=1)
        pct = pd.DataFrame(human_pct, columns=['human'])
        pct.reset_index(inplace=True)
        pct = pct[['video_id', 'start', 'human']].copy()
        return pct
