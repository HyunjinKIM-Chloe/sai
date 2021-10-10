import freq_dataset as freq
import cosine_similarity as cos


if __name__ == '__main__':
    sql = freq.MakeFreqDataset()
    sim = cos.CosineSimilarity()
    cursor, engine, db = sql.connect_sql()

    # 테이블 create
    # sql.create_frequency_tb(cursor)
    # sql.create_similarity_tb(cursor)

    # tracks_df = sql.read_db()
    # print(tracks_df['uri'].tolist())
    # print(tracks_df)

    # freq_df = sql.make_frequency_df()
    # sql.save_sql(engine, freq_df, 'frequency', 'append')
    # print(freq_df.columns)
    # print(len(freq_df))

    sim_df = sim.make_similarity_df()
    sql.save_sql(engine, sim_df, 'similarity', 'append')
    print(sim_df.columns)
    print(len(sim_df))


