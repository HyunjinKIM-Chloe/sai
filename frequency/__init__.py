import sys
import warnings
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
            arg2 = int(sys.argv[2])
            arg3 = int(sys.argv[3])
            freq_df = sql.make_frequency_df(arg2, arg3)
            print(f"Total {len(freq_df)} songs calculation completed!")
            print(freq_df.head())
            # sql.save_sql(engine, freq_df, 'frequency', 'append')
        elif arg1 == 'similarity':
            sim_df = sim.make_similarity_df()
            print(f"Total {len(sim_df)} songs calculation completed!")
            print(sim_df.head())
            # sql.save_sql(engine, sim_df, 'similarity', 'append')
        else:
            print('ERROR, NO ARGUMENTS')

    except:
        print('ERROR, NO ARGUMENTS')
