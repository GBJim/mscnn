data_info = {}




#Jackson Hole Data:
info = {}
info["name"] = "jacksonhole.mp4"
info["data_path"] = "/root/data/data-jacksonhole"
info["img_path"] = info["data_path"] + "/images"
info["training"] =  {"sets":[0], "videos":[0] , "start_frame":0, "end_frame":1080, "frame_stride":1}
info["testing"] = {"sets":[0], "videos":[0] , "start_frame":0, "end_frame":1080, "frame_stride":30}
data_info["jackson"] = info.copy()




#Chruch Street Data:
info = {}
info["name"] = "chruch_street.mp4"
info["data_path"] = "/root/data/data-chruch_street"
info["json_annotation"] = info["data_path"] + "/annotations.json"

info["img_path"] = info["data_path"] + "/images" 
info["training"] =  {"sets":[0], "videos":[0] , "start_frame":None, "end_frame":None, "frame_stride":1}  #temporary
info["testing"] = {"sets":[1], "videos":[0] , "start_frame":None, "end_frame":None, "frame_stride":30}
data_info["chruch_street"] = info.copy()



#Include other data like above