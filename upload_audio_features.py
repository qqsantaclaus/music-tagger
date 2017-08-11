import pymysql
import audio_learning as al
import json

learner = al.AudioLearner()
pairs = learner.feature(load = True)
db = pymysql.connect("595f58f641420.gz.cdb.myqcloud.com", "cdb_outerroot", "mini123456", "tingwen", 5880, use_unicode=True, charset="utf8")
cursor= db.cursor()
for (sid, feature) in pairs:
    try:
        print sid
        cursor.execute(
        '''REPLACE INTO
            songfeature (song_id, feature)
            VALUES (%s, %s)''', (sid, json.dumps(list(feature)))) 
        db.commit()
    except Exception, e:
        db.rollback()
        print("insert database error", e)
db.close()