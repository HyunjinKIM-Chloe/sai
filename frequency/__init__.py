import sys
import warnings
import pandas as pd
import freq_dataset as freq
import cosine_similarity as cos
warnings.filterwarnings(action='ignore')
sql = freq.MakeFreqDataset()
sim = cos.CosineSimilarity()
cursor, engine, db = sql.connect_sql()

if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
        if arg1 == 'frequency':
            try:
                arg2 = int(sys.argv[2])
                arg3 = int(sys.argv[3])
                freq_df = sql.make_frequency_df(arg2, arg3)
            except:
                freq_df = sql.make_frequency_df()
            print(f"Total {len(freq_df)} songs calculation completed!")
            freq_df.to_csv(f"frequency_{len(freq_df)}.csv", encoding='utf-8-sig')
            print(freq_df.head())
            sql.save_sql(engine, freq_df, 'frequency', 'append')
        elif arg1 == 'similarity':
            sim_df = sim.make_similarity_df()
            print(f"Total {len(sim_df)} songs calculation completed!")
            print(sim_df.head())
            sql.save_sql(engine, sim_df, 'similarity', 'append')
        else:
            print('ERROR, NO ARGUMENTS')

    except:
        freq_df.to_csv(f"frequency_{len(freq_df)}.csv", encoding='utf-8-sig')
        print('ERROR, NO ARGUMENTS')
