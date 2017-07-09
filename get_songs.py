import json
import urllib

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

for item in data:
    sid = item["id"]
    link = item["songLink"]
    urllib.urlretrieve(link, "/music/songs/"+sid+"/"+sid.mp3")
    with open("/music/songs/"+sid+"/info.json") as info_file:
	json.dump(item, info_file)
