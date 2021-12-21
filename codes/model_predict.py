import freq_dataset as fd
import pandas as pd
import yaml
import glob
import joblib
import warnings
warnings.simplefilter("ignore")


class Preprocessing(fd.MakeFreqDataset):
    def __init__(self):
        super().__init__()
        with open('../config/config.yaml') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.local_path = conf['local_path']  # path for local save
        self.model_path = conf['model_path']
        self.file_path = conf['file_path']
        self.label_ls = conf['label_ls']

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

    def predict_check_result(self, filename=None, start=None, end=None):
        result_df = self.make_predict_df(filename, start, end)
        count_df = result_df.groupby('name').sum() / len(self.model_ls)
        for name, item in count_df.iterrows():
            print(f"'{name}' 은 {round(max(item), 2) * 100}% 의 확률로"
                  f" {item.index[item.tolist().index(max(item))]}로 추정됩니다.")
        return count_df
