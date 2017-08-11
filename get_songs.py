import json
import urllib
import os
import requests

'''
json 

[
    {
        "id":"",         
        "songLink":"",     
        "lrcLink":"",       
        "songPicSmall":"",  
        "songName":"",      
        "tag":[ "","",""]        
    },
    ...
]
'''
json_file = "songs.json"

with open(json_file) as data_file:    
    data = json.load(data_file)

err_songs = []

for item in data:
    try:
        sid = item["id"]
        print sid
        link = "http://tingapi.ting.baidu.com/v1/restserver/ting?method=baidu.ting.song.play&songid="+sid
        directory = "./music/songs/"+sid
        if not os.path.exists(directory):
            os.makedirs(directory)
        response = json.loads(requests.get(link,verify=False, auth=('user', 'pass')).text)
        file_link =response['bitrate']['file_link']
        print item["songName"], response["songinfo"]["title"]
        print file_link
        urllib.urlretrieve(file_link, directory+"/"+sid+".mp3")
        # response2 = requests.get(file_link,verify=False, auth=('user', 'pass'))
        # with open(directory+"/"+sid+".mp3",'wb') as fout:
        #     fout.write(response2.text)
        with open(directory+"/info.json", "w") as info_file:
            json.dump(item, info_file)
    except Exception, e:
        err_songs.append(item)
print "====================== End of Downloading ======================="
print err_songs