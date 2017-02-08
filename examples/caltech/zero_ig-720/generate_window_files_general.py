import json
import os
import numpy as np
from data_info import data_info
import argparse
import sys
import cv2
from PIL import Image
import glob
from natsort import natsorted


#Public functions:
#For boxes with certain Label
def label_filter(box, label="person"):
    return box['lbl'] == label

#For boxes with a specified boundry, the default values arefrom 
def boundry_filter(box, bnds = {'xmin':5, 'ymin':5, 'xmax':635, 'ymax':475}):
    x1 = box['pos'][0]
    y1 = box['pos'][1]
    width = box['pos'][2]
    height = box['pos'][3]
    x2 = x1 + width
    y2 = y1 + height

    validity =  x1 >= bnds['xmin'] and \
                x2 <= bnds['xmax'] and \
                y1 >= bnds['ymin'] and \
                y2 <= bnds['ymax'] 

    return validity

#For boxes higher than a speifcied height
def height_filter(box, height_range = {'min':50, 'max': float('inf')}):
    height = box['pos'][3]
    validity = height >= height_range['min'] and \
               height < height_range['max']
    return validity

#For boxes more visible than a speifcied range
def visibility_filter(box, visible_range = {'min': 0.65, 'max': float('inf')}):
    occluded = box['occl']

    #A dirty condition to deal with the ill-formatted data.
    if occluded == 0 or \
       not hasattr(box['posv'], '__iter__') or \
       all([v==0 for v in box['posv']]):

        visiable_ratio = 1

    else:
        width = box['pos'][2]
        height = box['pos'][3]
        area = width * height   

        visible_width = box['posv'][2]
        visible_height = box['posv'][3]
        visible_area = visible_width * visible_height

        visiable_ratio = visible_area / area



    validity = visiable_ratio  >= visible_range['min'] and \
           visiable_ratio  <= visible_range['max']

    return validity


    height = box['pos'][3]
    validity = height >= height_range['min'] and \
               height < height_range['max']
    return validity




def reasonable_filter(box):
    label = "person"
    validity = box['lbl'] == 'person' and\
               boundry_filter(box) and\
               height_filter(box) and \
               visibility_filter(box)

    return validity


def load_annotations(anno_path, new_anno_path):
        
       
        assert os.path.exists(anno_path), \
                'Annotation path does not exist.: {}'.format(anno_path)
        annotation = json.load(open(anno_path))
            
      
        assert os.path.exists(new_anno_path), \
                'Annotation path does not exist.: {}'.format(new_anno_path)
        new_anno = json.load(open(new_anno_path))
        
        replacing_count = 0
        
                                 
      
        box_filter = reasonable_filter
        
        
        for set_num, set_anno in new_anno.items():
            for v_num, v_anno in set_anno.items():
                for frame_name, new_boxes in v_anno["frames"].items():
                    old_boxes = annotation[set_num][v_num]["frames"] .get(frame_name, [])
                    old_boxes = [old_box for old_box in old_boxes if box_filter(old_box)]
                    #annotation[set_num][v_num]["frames"][frame_name] = v_anno["frames"][frame_name]
                    merged_boxes = merge_boxes(old_boxes, new_boxes)
                    
                    if merged_boxes:
                        replacing_count += 1
                        
                    annotation[set_num][v_num]["frames"][frame_name] = merged_boxes
                        
                    
                    
                    
        
        
        print("{} frames of annotation are merged with new annotaions 10X".format(replacing_count))
        return annotation


#Generate the training annotations for CalTech Dataset
def generate_window_file(annotations, ouput_path, dataset="train"):
    pass

   


#Parse command line arguments input
def parse_args():
    parser = argparse.ArgumentParser(description='Generate window files from Vatic JSON annotations')
    parser.add_argument('--data', dest='data', help='dataset to select:{}'.format(data_info.keys()), type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
        
        
    return parser.parse_args()


def parse_annotation(data):
    annotation = json.load(open(data["json_annotation"]))

                
    return annotation        
            
            
    
            
def get_img_status(data):
     
    CHANNELS = 3    
    IMG_H = 480 
    IMG_W = 640
    
    
    img_path = data["img_path"]
    
    first_img = data["img_path"] + "/set00/V000/set00_V000_0.jpg"
    
    im=Image.open(first_img)
    img_w, img_h = im.size
    
    
    return (CHANNELS, img_h, img_w)
    
    
    
    
    
  
    
    




def generate_window_files(data, output_path, data_type):
    
    
    channels, img_h, img_w = get_img_status(data)
    
    annotation = parse_annotation(data)
    
    if data_type == "train":
        set_nums = data["training"]["sets"]
        start_frame = data["training"]["start_frame"] if data["training"]["start_frame"] else 0
        default_end_frame = data["training"]["end_frame"]
        frame_stride = data["training"]["frame_stride"]
    elif data_type == "test":
        set_nums = data["testing"]["sets"]
        start_frame = data["testing"]["start_frame"] if data["testing"]["start_frame"] else 0
        default_end_frame = data["testing"]["end_frame"]
        frame_stride = data["testing"]["frame_stride"]
    else:
        raise(Exception('My error!'))
    

    
    
    
    w = open(output_path, 'w')
    print("Start to wrtie file: {}".format(output_path))
    for set_num in set_nums:
        
          
        if default_end_frame is None:
            file_pattern = "{}/set0{}/V000/set0{}_V*".format(data["img_path"],set_num, set_num)
            print(file_pattern)
            file_list = natsorted(glob.glob(file_pattern))
            last_file = file_list[-1]
            end_frame =  int(last_file.split("_")[-1].split(".")[0])
        
        
        
        for frame in range(frame_stride-1, end_frame+1, frame_stride):
            img_path = data["img_path"] + "/set0{}/V000/set0{}_V000_{}.jpg".format(set_num, set_num,frame)
            ronis = []

            boxes = annotation[str(set_num)].get(str(frame), {}).values()
            w.write("# {}\n{}\n{}\n{}\n{}\n{}\n".format(frame, img_path, channels, img_h, img_w, len(boxes)))
            #print(boxes)
            for box in boxes:
                print(box)
                x1 = box['x1']
                y1 = box['y1']
                width = box['width']
                height = box['height']
                label = 1 if box['label'] == "person" else 0
                ignore = 1 if box['occluded'] or box['outside'] else 0

                w.write("{} {} {} {} {} {}\n".format(label, ignore, x1, y1, width, height))
            w.write("{}\n".format(len(ronis)))
        

        

    

    w.close()
    print("Finish Writing")
    
    
    

    

    
    

if __name__ == "__main__":
   


    
    args = parse_args()
    data = data_info[args.data]
    output_path = args.data + "_window_file.txt"
    
    
    #annotations = json.load(open(data["json_annotation"]))
    print(data)
    
    
    output_path = args.data + "_train_window_file.txt"
    generate_window_files(data, output_path,"train")
    output_path = args.data + "_test_window_file.txt"
    generate_window_files(data, output_path,"test")
       

    





