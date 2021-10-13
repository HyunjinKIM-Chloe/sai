import pandas as pd
import pymysql
import yaml
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()


class Query:
    def __init__(self):
        with open('../config/database.yaml') as f:
            conf = yaml.load(f)
        self.IP = conf['address']
        self.USER = conf['user']
        self.DB = conf['database_name']
        self.PW = conf['password']

    # db cursor 생성
    def connect_sql(self):
        engine = create_engine(f"mysql+mysqldb://{self.USER}:{self.PW}@{self.IP}/{self.DB}?charset=utf8", encoding='utf-8')

        conn = engine.connect()

        mydb = pymysql.connect(
            user=self.USER,
            passwd=self.PW,
            host=self.IP,
            db=self.DB,
            use_unicode=True,
            charset='utf8',
        )
        cursor = mydb.cursor(pymysql.cursors.DictCursor)
        return cursor, engine, mydb

    # db에 저장
    def save_sql(self, engine, data, table, option):
        data.to_sql(name=table, con=engine, if_exists=option, index=False)

    # db 특정 테이블 전체 데이터 읽어오기
    def read_db(self, table):
        cursor, engine, db = self.connect_sql()
        query = f"select * from {table}"
        cursor.execute(query)
        result = cursor.fetchall()
        result_df = pd.DataFrame(result)
        return result_df

    # db 테이블 생성
    def create_frequency_tb(self, cursor):
        try:
            qry_drop = (f'''
                        drop table {self.DB}.frequency;
                        ''')
            cursor.execute(qry_drop)
        except:
            pass

        qry_result = (f'''
            CREATE TABLE {self.DB}.frequency (
                id                           INT            NOT NULL    AUTO_INCREMENT COMMENT 'Surrogate Key', 
                track_id                     INT            NOT NULL    UNIQUE         COMMENT 'tracks 테이블 id', 
                chroma_stft_mean             FLOAT          NOT NULL, 
                chroma_stft_var              FLOAT          NOT NULL, 
                rms_mean                     FLOAT          NOT NULL, 
                rms_var                      FLOAT          NOT NULL, 
                spectral_centroid_mean       FLOAT          NOT NULL,
                spectral_centroid_var        FLOAT          NOT NULL,
                spectral_bandwidth_mean      FLOAT          NOT NULL,
                spectral_bandwidth_var       FLOAT          NOT NULL,
                rolloff_mean                 FLOAT          NOT NULL,
                rolloff_var                  FLOAT          NOT NULL,
                zero_crossing_rate_mean      FLOAT          NOT NULL,
                zero_crossing_rate_var       FLOAT          NOT NULL,
                harmony_mean                 FLOAT          NOT NULL,
                harmony_var                  FLOAT          NOT NULL,
                perceptr_mean                FLOAT          NOT NULL,
                perceptr_var                 FLOAT          NOT NULL,
                tempo                        FLOAT          NOT NULL,
                mfcc1_mean                   FLOAT          NOT NULL,
                mfcc1_var                    FLOAT          NOT NULL,
                mfcc2_mean                   FLOAT          NOT NULL,
                mfcc2_var                    FLOAT          NOT NULL,
                mfcc3_mean                   FLOAT          NOT NULL,
                mfcc3_var                    FLOAT          NOT NULL,
                mfcc4_mean                   FLOAT          NOT NULL,
                mfcc4_var                    FLOAT          NOT NULL,
                mfcc5_mean                   FLOAT          NOT NULL,
                mfcc5_var                    FLOAT          NOT NULL,
                mfcc6_mean                   FLOAT          NOT NULL,
                mfcc6_var                    FLOAT          NOT NULL,
                mfcc7_mean                   FLOAT          NOT NULL,
                mfcc7_var                    FLOAT          NOT NULL,
                mfcc8_mean                   FLOAT          NOT NULL,
                mfcc8_var                    FLOAT          NOT NULL,
                mfcc9_mean                   FLOAT          NOT NULL,
                mfcc9_var                    FLOAT          NOT NULL,
                mfcc10_mean                  FLOAT          NOT NULL,
                mfcc10_var                   FLOAT          NOT NULL,
                mfcc11_mean                  FLOAT          NOT NULL,
                mfcc11_var                   FLOAT          NOT NULL,
                mfcc12_mean                  FLOAT          NOT NULL,
                mfcc12_var                   FLOAT          NOT NULL,
                mfcc13_mean                  FLOAT          NOT NULL,
                mfcc13_var                   FLOAT          NOT NULL,
                mfcc14_mean                  FLOAT          NOT NULL,
                mfcc14_var                   FLOAT          NOT NULL,
                mfcc15_mean                  FLOAT          NOT NULL,
                mfcc15_var                   FLOAT          NOT NULL,
                mfcc16_mean                  FLOAT          NOT NULL,
                mfcc16_var                   FLOAT          NOT NULL,
                mfcc17_mean                  FLOAT          NOT NULL,
                mfcc17_var                   FLOAT          NOT NULL,
                mfcc18_mean                  FLOAT          NOT NULL,
                mfcc18_var                   FLOAT          NOT NULL,
                mfcc19_mean                  FLOAT          NOT NULL,
                mfcc19_var                   FLOAT          NOT NULL,
                mfcc20_mean                  FLOAT          NOT NULL,
                mfcc20_var                   FLOAT          NOT NULL,
                CONSTRAINT PK_place PRIMARY KEY (id, track_id)
            );
                ''')
        cursor.execute(qry_result)

    def create_similarity_tb(self, cursor):
        try:
            qry_drop = (f'''
                        drop table {self.DB}.similarity;
                        ''')
            cursor.execute(qry_drop)
        except:
            pass

        qry_result = (f'''
            CREATE TABLE {self.DB}.similarity (
                id                INT            NOT NULL    AUTO_INCREMENT COMMENT 'Surrogate Key', 
                track1            VARCHAR(45)    NOT NULL                   COMMENT '기준 곡', 
                track2            VARCHAR(45)    NOT NULL                   COMMENT '비교 곡', 
                similarity        FLOAT          NOT NULL                   COMMENT '코사인 유사도 (-1~1)', 
                CONSTRAINT PK_similarity PRIMARY KEY (id)
            );
                ''')
        cursor.execute(qry_result)