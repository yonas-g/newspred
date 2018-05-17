import sqlite3
import pandas as pd
import numpy as np
import os

def create_db_news(filename):
    con = sqlite3.connect(filename)
    # with con:
    #     cur = con.cursor()
    #     cur.execute("CREATE TABLE News(id INTEGER PRIMARY KEY, title TEXT, description TEXT, urlToImage TEXT, url TEXT, source TEXT, author TEXT, publishedAt TEXT)")
    #con.commit()

    con.close()


def save_db_news(filename, data):
    if not os.path.isfile(filename):
        con = sqlite3.connect(filename)
        last_index = 0
    else:
        con = sqlite3.connect(filename)
        cur = con.cursor()
        cursor = con.execute('SELECT max(id) FROM News')
        last_index = cursor.fetchone()[0]
    len_data = len(data)
    new_data = []
    for item in data:
        last_index +=1
        item['source_id'] = item['source']['id']
        item['source_name'] = item['source']['name']
        item['id'] = last_index
        item.pop('source', None)
        new_data.append(item)

    datapd = pd.DataFrame.from_records(new_data)

    with con:
        datapd.to_sql(name="News", con=con, if_exists="append", index=False)
    con.close()
