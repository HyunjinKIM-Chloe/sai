import pandas as pd
import pymysql
import yaml
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()


class Query:
    def __init__(self):
        with open('../config/config.yaml') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.IP = conf['address']
        self.USER = conf['user']
        self.DB = conf['database_name']
        self.PW = conf['password']

    # db cursor 생성
    def connect_sql(self):
        engine = create_engine(f"mysql+mysqldb://{self.USER}:{self.PW}@{self.IP}/{self.DB}?charset=utf8",
                               encoding='utf-8')

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
        print(f"Total {len(result_df)} rows")
        return result_df


if __name__ == '__main__':
    sql = Query()
    cursor, engine, db = sql.connect_sql()
    videos_df = sql.read_db('videos')
    print(videos_df[videos_df['id'] == 9]['uri'].iloc[0])

