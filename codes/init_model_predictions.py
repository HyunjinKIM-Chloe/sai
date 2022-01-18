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
            freq_df = mmp.predict_check_result(video_id=arg1, start=float(arg2), end=float(arg3))
            sql.save_sql(engine, freq_df, 'model_predictions', 'append')
            print(f"id-{arg1} video ({arg2}s to {arg3}s): {len(freq_df)} rows INSERT COMPLETED!")
        except:
            arg1 = int(sys.argv[1])
            freq_df = mmp.predict_check_result(video_id=arg1)
            sql.save_sql(engine, freq_df, 'model_predictions', 'append')
            print(f"id-{arg1} video: {len(freq_df)} rows INSERT COMPLETED!")

    except:
        print("Please check video_id!")
