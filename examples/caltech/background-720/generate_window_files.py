#Window files generator for background-MSCNN
#Enables MSCNN to take background images(maybe not?)

import json
import os
import numpy as np
from glob import glob
from natsort import natsorted
import re
DATA_PATH = "/root/data/caltech-pedestrian-dataset-converter/data"

ANNO_PATH = os.path.join(DATA_PATH, "annotations.json")
ANNO_PATH_10X = os.path.join(DATA_PATH, "new_anno_10x.json")
ANNO_PATH_1X = os.path.join(DATA_PATH, "new_anno.json")
IMG_DIR = os.path.join(DATA_PATH, "images")
IMG_FORMAT = 'jpg'

CHANNELS = 3 
IMG_H = 480 
IMG_W = 640


def get_overlap_area(box_a, box_b):
    x1_a, y1_a, width_a, height_a = box_a['pos']
    x1_b, y1_b, width_b, height_b = box_b['pos']

    x2_a = x1_a + width_a
    y2_a = y1_a + height_a
    x2_b = x1_b + width_b
    y2_b = y1_b + height_b

    #get the width and height of overlap rectangle
    overlap_width =  min(x2_a, x2_b) - max(x1_a, x1_b) 
    overlap_height = min(y2_a, y2_b) - max(y1_a, y1_b) 

    #If the width or height of overlap rectangle is negative, it implies that two rectangles does not overlap.
    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0
    
def get_IOU(box_a, box_b):

    overlap_area = get_overlap_area(box_a, box_b)

    #Union = A + B - I(A&B)
    area_a = box_a['pos'][2] * box_a['pos'][3]
    area_b = box_b['pos'][2] * box_b['pos'][3]
    union_area = area_a + area_b - overlap_area


    if overlap_area > 0 :
        return union_area / overlap_area
    else:
        return 0

def get_max_IOU(box, mateched_boxes):
    return max([(i, get_IOU(box, matched_box)) for matched_box in mateched_boxes],key=itemgetter(1))

def merge_boxes(old_boxes, new_boxes, IOU_thresh = 0.7):
    if len(old_boxes) > 0 and len(new_boxes) > 0:
        IOU_table = np.zeros((len(old_boxes), len(new_boxes)),dtype=float)
        merged_boxes = []
       
        #Fill the IOU values into IOU tables 
        for i, old_box in enumerate(old_boxes):
            for j, new_box in enumerate(new_boxes):
                IOU_table[i, j] = get_IOU(old_box, new_box)
        merge_count = 0
        #Filling old or new box into merged boxes
        for i, old_box in enumerate(old_boxes):
            #Find the best match of i-th old_box
            matched_index = np.argmax(IOU_table[i,:])
            merge_box = old_boxes[i]
                
            #Check if i-th old box is also the strongest match of the matched new box
            if i == np.argmax(IOU_table[:,matched_index]):
                merge_box['pos'] = new_boxes[matched_index]['pos']
                merge_count += 1
             
            merged_boxes.append(merge_box)
         
              
        #print("{} boxes are merged".format(merge_count))
        return merged_boxes
            
        
        
    else:
        return []
        











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
        
       #
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
    ignore_elements = []

    train_sets = ["set00","set01","set02","set03","set04", "set05"]
    test_sets = ["set06","set07","set08","set09","set10"]
    counter = 0


    window_file = open(ouput_path, "w")

    set_nums = train_sets if dataset == "train" else test_sets
    for set_num in set_nums:
        for v_num, v_anno in sorted(annotations[set_num].items(), key=lambda x: int(x[0][1:])):
            image_list = glob("{}/{}_{}*".format(IMG_DIR, set_num, v_num))
            image_list = natsorted(image_list)
            #print(len(image_list))
            #print(image_list[0:5])
            
            for image in [os.path.basename(image) for image in  image_list]:
                frame_name = image.split("_")[-1][:-4]  
                bboxes = v_anno["frames"].get(frame_name, [])
                print(len(bboxes))
                bboxes = [bbox for bbox in bboxes if reasonable_filter(bbox)]
            
            
     
                img_name = "{}_{}_{}.{}".format(set_num, v_num, frame_name, IMG_FORMAT)           
                img_path = os.path.join(IMG_DIR, img_name)
            
                window_file.write("# {}\n{}\n{}\n{}\n{}\n{}\n".format(counter, img_path, CHANNELS, IMG_H, IMG_W, len(bboxes)))
                for bbox in bboxes:
                    label = 1 if bbox['lbl'] == 'person' else 0
                    ignore = bbox['occl']
                    x1, y1, w, h = [ int(round(value)) for value in bbox['pos']]
                    x2 = x1 + w
                    y2 = y1 + h
                    window_file.write("{} {} {} {} {} {}\n".format(label, ignore, x1, y1, x2 - 1, y2 - 1))
                window_file.write("{}\n".format(len(ignore_elements)))
            counter += 1
    window_file.close()


if __name__ == "__main__":
    annotations_10X = load_annotations(ANNO_PATH, ANNO_PATH_10X)
    annotations_1X = json.load(open(ANNO_PATH_1X))




    output_path = "./10X_mscnn_window_file_caltech_train.txt"
    dataset = "train"
    generate_window_file(annotations_10X, output_path, dataset)
    output_path = "./1X_mscnn_window_file_caltech_test.txt"
    dataset = "test"
    generate_window_file(annotations_1X, output_path, dataset)






