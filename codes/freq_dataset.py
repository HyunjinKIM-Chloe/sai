import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm


class Preprocessing():
    def __init__(self):
        self.n_fft = 2048
        self.hop_length = 512
        self.frame_length = 2048

    def read_file(self, url):
        y, sr = librosa.load(url)
        return y, sr

    def calc_duration(self, path, y, sr):
        try:
            audio = wave.open(path)
            frames = audio.getnframes()
            rate = audio.getframerate()
            duration = frames / float(rate)
        except:
            # y, sr = librosa.load(p, sr=16000)
            duration = len(y)/sr
        return duration

    def make_chroma(self, y, sr):
        result = librosa.feature.chroma_stft(y, sr, n_fft=self.n_fft, hop_length=self.hop_length)
        chroma_stft_mean = np.mean(result)
        chroma_stft_var = np.var(result)
        return chroma_stft_mean, chroma_stft_var

    def make_rms(self, y):
        result = librosa.feature.rms(y, frame_length=self.frame_length, hop_length=self.hop_length)
        rms_mean = np.mean(result)
        rms_var = np.var(result)
        return rms_mean, rms_var

    def make_spectral_centroid(self, y, sr):
        result = librosa.feature.spectral_centroid(y, sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_centroid_mean = np.mean(result)
        spectral_centroid_var = np.var(result)

        return spectral_centroid_mean, spectral_centroid_var

    def make_spectral_bandwidth(self, y, sr):
        result = librosa.feature.spectral_bandwidth(y, sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_bandwidth_mean = np.mean(result)
        spectral_bandwidth_var = np.var(result)
        return spectral_bandwidth_mean, spectral_bandwidth_var

    def make_spectral_rolloff(self, y, sr):
        result = librosa.feature.spectral_rolloff(y, sr, n_fft=self.n_fft, hop_length=self.hop_length)
        rolloff_mean = np.mean(result)
        rolloff_var = np.var(result)
        return rolloff_mean, rolloff_var

    def make_zero_crossing_rate(self, y):
        result = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length)
        zero_crossing_rate_mean = np.mean(result)
        zero_crossing_rate_var = np.var(result)
        return zero_crossing_rate_mean, zero_crossing_rate_var

    def har_per(self, y):
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        har_mean = np.mean(y_harmonic)
        har_var = np.var(y_harmonic)
        per_mean = np.mean(y_percussive)
        per_var = np.var(y_percussive)

        return har_mean, har_var, per_mean, per_var

    def make_tempo(self, y, sr):
        result = librosa.beat.tempo(y, sr, hop_length=self.hop_length)
        return result

    def make_mfcc(self, y, sr):
        # n_mfcc는 20
        result = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(result, axis=1)
        mfcc_var = np.var(result, axis=1)
        return mfcc_mean, mfcc_var


class MakeFreqDataset(Preprocessing):
    def __init__(self):
        super().__init__()

    def make_frequency_df(self, tracks_df, start=None, end=None):
        ls_chroma_stft_mean = []
        ls_chroma_stft_var = []
        ls_rms_mean = []
        ls_rms_var = []
        ls_spectral_centroid_mean = []
        ls_spectral_centroid_var = []
        ls_spectral_bandwidth_mean = []
        ls_spectral_bandwidth_var = []
        ls_rolloff_mean = []
        ls_rolloff_var = []
        ls_zero_crossing_rate_mean = []
        ls_zero_crossing_rate_var = []
        ls_harmony_mean = []
        ls_harmony_var = []
        ls_perceptr_mean = []
        ls_perceptr_var = []
        ls_tempo = []
        ls_mfcc1_mean = []
        ls_mfcc1_var = []
        ls_mfcc2_mean = []
        ls_mfcc2_var = []
        ls_mfcc3_mean = []
        ls_mfcc3_var = []
        ls_mfcc4_mean = []
        ls_mfcc4_var = []
        ls_mfcc5_mean = []
        ls_mfcc5_var = []
        ls_mfcc6_mean = []
        ls_mfcc6_var = []
        ls_mfcc7_mean = []
        ls_mfcc7_var = []
        ls_mfcc8_mean = []
        ls_mfcc8_var = []
        ls_mfcc9_mean = []
        ls_mfcc9_var = []
        ls_mfcc10_mean = []
        ls_mfcc10_var = []
        ls_mfcc11_mean = []
        ls_mfcc11_var = []
        ls_mfcc12_mean = []
        ls_mfcc12_var = []
        ls_mfcc13_mean = []
        ls_mfcc13_var = []
        ls_mfcc14_mean = []
        ls_mfcc14_var = []
        ls_mfcc15_mean = []
        ls_mfcc15_var = []
        ls_mfcc16_mean = []
        ls_mfcc16_var = []
        ls_mfcc17_mean = []
        ls_mfcc17_var = []
        ls_mfcc18_mean = []
        ls_mfcc18_var = []
        ls_mfcc19_mean = []
        ls_mfcc19_var = []
        ls_mfcc20_mean = []
        ls_mfcc20_var = []
        ls_duration = []

        # tracks_df = self.read_db('tracks')
        for url in tqdm(tracks_df['path'].tolist()[start:end]):
            y, sr = self.read_file(url)
            duration = self.calc_duration(url, y, sr)
            chroma_stft_mean, chroma_stft_var = self.make_chroma(y, sr)
            rms_mean, rms_var = self.make_rms(y)
            spectral_centroid_mean, spectral_centroid_var = self.make_spectral_centroid(y, sr)
            spectral_bandwidth_mean, spectral_bandwidth_var = self.make_spectral_bandwidth(y, sr)
            rolloff_mean, rolloff_var = self.make_spectral_rolloff(y, sr)
            zero_crossing_rate_mean, zero_crossing_rate_var = self.make_zero_crossing_rate(y)
            har_mean, har_var, per_mean, per_var = self.har_per(y)
            tempo = self.make_tempo(y, sr)
            mfcc_mean, mfcc_var = self.make_mfcc(y, sr)

            ls_chroma_stft_mean.append(chroma_stft_mean)
            ls_chroma_stft_var.append(chroma_stft_var)
            ls_rms_mean.append(rms_mean)
            ls_rms_var.append(rms_var)
            ls_spectral_centroid_mean.append(spectral_centroid_mean)
            ls_spectral_centroid_var.append(spectral_centroid_var)
            ls_spectral_bandwidth_mean.append(spectral_bandwidth_mean)
            ls_spectral_bandwidth_var.append(spectral_bandwidth_var)
            ls_rolloff_mean.append(rolloff_mean)
            ls_rolloff_var.append(rolloff_var)
            ls_zero_crossing_rate_mean.append(zero_crossing_rate_mean)
            ls_zero_crossing_rate_var.append(zero_crossing_rate_var)
            ls_harmony_mean.append(har_mean)
            ls_harmony_var.append(har_var)
            ls_perceptr_mean.append(per_mean)
            ls_perceptr_var.append(per_var)
            ls_tempo.append(tempo[0])
            ls_mfcc1_mean.append(mfcc_mean[0])
            ls_mfcc1_var.append(mfcc_var[0])
            ls_mfcc2_mean.append(mfcc_mean[1])
            ls_mfcc2_var.append(mfcc_var[1])
            ls_mfcc3_mean.append(mfcc_mean[2])
            ls_mfcc3_var.append(mfcc_var[2])
            ls_mfcc4_mean.append(mfcc_mean[3])
            ls_mfcc4_var.append(mfcc_var[3])
            ls_mfcc5_mean.append(mfcc_mean[4])
            ls_mfcc5_var.append(mfcc_var[4])
            ls_mfcc6_mean.append(mfcc_mean[5])
            ls_mfcc6_var.append(mfcc_var[5])
            ls_mfcc7_mean.append(mfcc_mean[6])
            ls_mfcc7_var.append(mfcc_var[6])
            ls_mfcc8_mean.append(mfcc_mean[7])
            ls_mfcc8_var.append(mfcc_var[7])
            ls_mfcc9_mean.append(mfcc_mean[8])
            ls_mfcc9_var.append(mfcc_var[8])
            ls_mfcc10_mean.append(mfcc_mean[9])
            ls_mfcc10_var.append(mfcc_var[9])
            ls_mfcc11_mean.append(mfcc_mean[10])
            ls_mfcc11_var.append(mfcc_var[10])
            ls_mfcc12_mean.append(mfcc_mean[11])
            ls_mfcc12_var.append(mfcc_var[11])
            ls_mfcc13_mean.append(mfcc_mean[12])
            ls_mfcc13_var.append(mfcc_var[12])
            ls_mfcc14_mean.append(mfcc_mean[13])
            ls_mfcc14_var.append(mfcc_var[13])
            ls_mfcc15_mean.append(mfcc_mean[14])
            ls_mfcc15_var.append(mfcc_var[14])
            ls_mfcc16_mean.append(mfcc_mean[15])
            ls_mfcc16_var.append(mfcc_var[15])
            ls_mfcc17_mean.append(mfcc_mean[16])
            ls_mfcc17_var.append(mfcc_var[16])
            ls_mfcc18_mean.append(mfcc_mean[17])
            ls_mfcc18_var.append(mfcc_var[17])
            ls_mfcc19_mean.append(mfcc_mean[18])
            ls_mfcc19_var.append(mfcc_var[18])
            ls_mfcc20_mean.append(mfcc_mean[19])
            ls_mfcc20_var.append(mfcc_var[19])
            ls_duration.append(duration)

        result_df = pd.DataFrame({
            "name": tracks_df['name'].tolist()[start:end],
            "path": tracks_df['path'].tolist()[start:end],
            # "start": tracks_df['start'].tolist()[start:end],
            # "end": tracks_df['end'].tolist()[start:end],
            # "label": [name.split('/')[-2] for name in tracks_df['path'].tolist()[start:end]],
            "duration": ls_duration,
            "chroma_stft_mean": ls_chroma_stft_mean,
            "chroma_stft_var": ls_chroma_stft_var,
            "rms_mean": ls_rms_mean,
            "rms_var": ls_rms_var,
            "spectral_centroid_mean": ls_spectral_centroid_mean,
            "spectral_centroid_var": ls_spectral_centroid_var,
            "spectral_bandwidth_mean": ls_spectral_bandwidth_mean,
            "spectral_bandwidth_var": ls_spectral_bandwidth_var,
            "rolloff_mean": ls_rolloff_mean,
            "rolloff_var": ls_rolloff_var,
            "zero_crossing_rate_mean": ls_zero_crossing_rate_mean,
            "zero_crossing_rate_var": ls_zero_crossing_rate_var,
            "harmony_mean": ls_harmony_mean,
            "harmony_var": ls_harmony_var,
            "perceptr_mean": ls_perceptr_mean,
            "perceptr_var": ls_perceptr_var,
            "tempo": ls_tempo,
            "mfcc1_mean": ls_mfcc1_mean,
            "mfcc1_var": ls_mfcc1_var,
            "mfcc2_mean": ls_mfcc2_mean,
            "mfcc2_var": ls_mfcc2_var,
            "mfcc3_mean": ls_mfcc3_mean,
            "mfcc3_var": ls_mfcc3_var,
            "mfcc4_mean": ls_mfcc4_mean,
            "mfcc4_var": ls_mfcc4_var,
            "mfcc5_mean": ls_mfcc5_mean,
            "mfcc5_var": ls_mfcc5_var,
            "mfcc6_mean": ls_mfcc6_mean,
            "mfcc6_var": ls_mfcc6_var,
            "mfcc7_mean": ls_mfcc7_mean,
            "mfcc7_var": ls_mfcc7_var,
            "mfcc8_mean": ls_mfcc8_mean,
            "mfcc8_var": ls_mfcc8_var,
            "mfcc9_mean": ls_mfcc9_mean,
            "mfcc9_var": ls_mfcc9_var,
            "mfcc10_mean": ls_mfcc10_mean,
            "mfcc10_var": ls_mfcc10_var,
            "mfcc11_mean": ls_mfcc11_mean,
            "mfcc11_var": ls_mfcc11_var,
            "mfcc12_mean": ls_mfcc12_mean,
            "mfcc12_var": ls_mfcc12_var,
            "mfcc13_mean": ls_mfcc13_mean,
            "mfcc13_var": ls_mfcc13_var,
            "mfcc14_mean": ls_mfcc14_mean,
            "mfcc14_var": ls_mfcc14_var,
            "mfcc15_mean": ls_mfcc15_mean,
            "mfcc15_var": ls_mfcc15_var,
            "mfcc16_mean": ls_mfcc16_mean,
            "mfcc16_var": ls_mfcc16_var,
            "mfcc17_mean": ls_mfcc17_mean,
            "mfcc17_var": ls_mfcc17_var,
            "mfcc18_mean": ls_mfcc18_mean,
            "mfcc18_var": ls_mfcc18_var,
            "mfcc19_mean": ls_mfcc19_mean,
            "mfcc19_var": ls_mfcc19_var,
            "mfcc20_mean": ls_mfcc20_mean,
            "mfcc20_var": ls_mfcc20_var})

        return result_df


if __name__ == '__main__':
    freq = MakeFreqDataset()
    s_path = "../sounds/"
    o_path = list(set(glob.glob(f'{s_path}*/*/*')))  # - set(final_df['path']))
    path_ls = [file for file in o_path if file.endswith(".mp3") or file.endswith(".wav")]
    name_ls = [name.split('/')[-1][:-4] for name in path_ls]
    label_ls = [name.split('/')[-3] for name in path_ls]
    info_df = pd.DataFrame()
    info_df['path'] = path_ls
    info_df['label'] = label_ls
    info_df['name'] = name_ls
    result_df = freq.make_frequency_df(tracks_df=info_df[:3])
    result_df.to_csv("../files/freq_dataset_1113.csv", encoding='utf-8-sig')
