import model_predict as mp
import access_db as db
import sys
import warnings
warnings.filterwarnings(action='ignore')
mmp = mp.ModelPredict()
sql = db.Query()
cursor, engine, db = sql.connect_sql()

if __name__ == '__main__':
    try:
        try:
            arg1 = int(sys.argv[1])
            arg2 = sys.argv[2]
            arg3 = sys.argv[3]
            freq_df = mmp.save_frequency_df(video_id=arg1, start=int(arg2), end=int(arg3))
            sql.save_sql(engine, freq_df, 'sound_features', 'append')
            print(f"id-{arg1} video: {len(freq_df)} rows INSERT COMPLETED!")
        except:
            arg1 = int(sys.argv[1])
            freq_df = mmp.save_frequency_df(video_id=arg1)
            sql.save_sql(engine, freq_df, 'sound_features', 'append')
            print(f"id-{arg1} video: {len(freq_df)} rows INSERT COMPLETED!")

    except:
        print("Please check video_id!")
