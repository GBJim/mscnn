#This script does not have the concenpt of videos 



from __future__ import division
import os
import math
import numpy as np
import json
from os import listdir
from os.path import isfile, join
import sys
import glob
import cv2
import argparse
import re
import time
from scipy.misc import imread
from data_info import data_info
from matplotlib import pyplot as plt
sys.path.insert(0, "../../../")
from nms.gpu_nms import gpu_nms
sys.path.insert(0, "../../../python/")
import caffe


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff



def parse_args():
    """
    Parse input arguments
    """
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description='Test a MSCNN network')
    
    parser.add_argument('--data', dest='data', help='dataset to test', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    
    parser.add_argument('--viz', dest='viz',help="Whether to visualize detection and store in output directory", action="store_true", default = False)
    

    
    parser.add_argument('--net', dest='prototxt',
                        help='prototxt file defining the network',
                        default=os.path.join(cwd,'mscnn_deploy.prototxt'), type=str)
    parser.add_argument('--weights', dest='caffemodel',
                        help='model to test',
                        default=os.path.join(cwd,'mscnn_caltech_train_2nd_iter_25000.caffemodel')\
                        , type=str)

    parser.add_argument('--do_bb_norm', dest='do_bb_norm',help="Whether to denormalize the box with std or means.\
    Author's pretrained model does not need this. ",
                default=True , type=bool)
    
    
    parser.add_argument('--height', dest='height',help="Decide the resizing height of input model and images",
                default=720 , type=int)
    
    parser.add_argument('--output', dest='output',  help='model to test', type=str)
    
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
   



def filter_proposals(proposals, threshold=-10):
    #Bug 1 Fixed
    keeps = (proposals[:, -1] >= threshold) & (proposals[:, 2] != 0) & (proposals[:, 3] != 0)
 
    return keeps




 

def im_normalize(im, target_size, mu=[104, 117, 123] ):
    n_im = cv2.resize(im, target_size).astype(np.float32)
    
    #Substracts mu from testing-BGR image
    n_im -= mu
    
    #print(im.shape)
    n_im = np.swapaxes(n_im, 1,2)
    n_im = np.swapaxes(n_im, 0,1)
    n_im = np.array([n_im])
    #print(n_im.shape)
    #print(n_im.shape)

  

    
    return n_im


def bbox_denormalize(bbox_pred, proposals, ratios, orgW, orgH):
    
    bbox_means = [0, 0, 0, 0]
    bbox_stds = [0.1, 0.1, 0.2, 0.2]

    if args.do_bb_norm:
        bbox_pred *= bbox_stds 
        bbox_pred += bbox_means


    
    
    ctr_x = proposals[:,0]+0.5*proposals[:,2]
    ctr_y = proposals[:,1]+0.5*proposals[:,3]





    tx = bbox_pred[:,0] *proposals[:,2] + ctr_x
    ty = bbox_pred[:,1] *proposals[:,3] + ctr_y

    tw = proposals[:,2] * np.exp(bbox_pred[:,2])
    th = proposals[:,3] * np.exp(bbox_pred[:,3])

    #Fix Bug 2
    tx -= tw/2 
    ty -= th/2
    tx /= ratios[0] 
    tw /= ratios[0]
    ty /= ratios[1] 
    th /= ratios[1]

    tx[tx < 0] = 0
    ty[ty < 0] = 0
    #Fix Bug 3
    tw[tw > (orgW - tx)] = (orgW - tx[tw > (orgW - tx)])
    th[th > (orgH - ty)] = (orgH - ty[th > (orgH - ty)])
    
    new_boxes = np.hstack((tx[:, None], ty[:, None], tw[:, None], th[:, None])).astype(np.float32).reshape((-1, 4)) #suspecious
    
    return new_boxes


def get_confidence(cls_pred):
    exp_score = np.exp(cls_pred)
    sum_exp_score = np.sum(exp_score, 1)
    confidence = exp_score[:, 1] / sum_exp_score
    
    return confidence


    


#mu is the mean of BGR
def im_detect(net, file_path, target_size= (960, 720)):
       
 
    im = cv2.imread(file_path)
    
    orgH, orgW, _ = im.shape
    ratios = (target_size[0]/orgW, (target_size[1]/orgH ))
    im = im_normalize(im, target_size)
    
    #Feedforward
    net.blobs['data'].data[...] = im 
    output = net.forward()
    
    bbox_pred = output['bbox_pred']
    proposals = output['proposals_score'].reshape((-1,6))[:,1:]  #suspecious
    
    proposals[:,2] -=   proposals[:,0]
    proposals[:,3] -=   proposals[:,1]
    cls_pred = output['cls_pred']
    
    
    keeps = filter_proposals(proposals)
    bbox_pred =  bbox_pred[keeps]
    cls_pred = cls_pred[keeps]
    proposals = proposals[keeps]
    
    pedestrian_boxes = bbox_pred[:,4:8]
    boxes = bbox_denormalize(pedestrian_boxes, proposals, ratios, orgW, orgH)

    #Denormalize the confidence 
    
    confidence = get_confidence(cls_pred)
   
    
    return confidence, boxes
    




def nms(dets, thresh):
    
    if dets.shape[0] == 0:
        return []
    new_dets = np.copy(dets)
    new_dets[:,2] += new_dets[:,0]
    new_dets[:,3] += new_dets[:,1]
   
    return gpu_nms(new_dets, thresh, device_id=GPU_ID)
  




def write_testing_results_file(net, data):



    # The follwing nested fucntions are for smart sorting
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]
    
    
    

    def insert_frame(target_frames, file_path,start_frame, frame_stride, end_frame):
        file_name = file_path.split("/")[-1]
        set_num, v_num, frame_num = file_name[:-4].split("_")
        condition = int(frame_num) >= start_frame and (int(frame_num)+1) % frame_stride == 0 and int(frame_num) < end_frame
        print(frame_num,start_frame, frame_stride, end_frame, condition)

        if condition:
            target_frames.setdefault(set_num,{}).setdefault(v_num,[]).append(file_path)
            return 1
        else:
            return 0

 
    
    def detect(file_path,  NMS_THRESH = 0.3):
        if args.height == 720:
            target_size = (960, 720)
        elif args.height == 480:
            target_size = (640, 480)
            
     

        confidence, bboxes = im_detect(net, file_path, target_size)
    
       
        dets = np.hstack((bboxes,confidence[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        print("{} Bboxes".format(len(keep)))
        return dets[keep, :]


    def get_target_frames(image_set_list, data):
        image_path = data["img_path"]
        start_frame = data["testing"]["start_frame"]
        end_frame = data["testing"]["end_frame"]
        frame_stride = data["testing"]["frame_stride"]
        
        if start_frame is None:
            start_frame = 0
        
        target_frames = {}
        total_frames = 0 
        for set_num in image_set_list:
            file_pattern = "{}/set{}/V000/set{}_V*".format(image_path,set_num,set_num)
            #print(file_pattern)
            print(file_pattern)
            file_list = sorted(glob.glob(file_pattern), key=natural_keys)
            
            if end_frame is None:
                last_file = file_list[-1]
                end_frame =  int(last_file.split("_")[-1].split(".")[0])
                
            #print(file_list)
            for file_path in file_list:
                total_frames += insert_frame(target_frames, file_path, start_frame, frame_stride, end_frame)

        return target_frames, total_frames 
    
    

    def detection_to_file(target_path, v_num, file_list, detect,total_frames, current_frames, max_proposal=100, thresh=0):
        timer = Timer()
        w = open("{}/{}.txt".format(target_path, v_num), "w")
        for file_index, file_path in enumerate(file_list):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            frame_num = str(int(frame_num) +1)

            timer.tic()
            dets = detect(file_path)

            timer.toc()

            print('Detection Time:{:.3f}s on {}  {}/{} images'.format(timer.average_time,\
                                                   file_name ,current_frames+file_index+1 , total_frames))


            inds = np.where(dets[:, -1] >= thresh)[0]   
            
            if args.viz:
                viz_path = os.path.join(target_path, "images")
                if not os.path.exists( viz_path):
                    os.mkdir(viz_path)
                render_image(file_path, dets, inds, viz_path)
            
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                
                
                #Fix bug 6
                x = bbox[0]
                y = bbox[1] 
                width = bbox[2] 
                length =  bbox[3]
                if score*100 > 70:
                    print("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score*100))
                    
                if length > 50:            
                    w.write("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score*100))
                else:
                    print("Bbox too small")


        w.close()
        print("Evalutaion file {} has been writen".format(w.name))   
        return file_index + 1








    image_set_list = [ str(set_num).zfill(2) for set_num in data["testing"]["sets"]]
    
    target_frames, total_frames = get_target_frames(image_set_list,  data)
    print(target_frames)
    print(total_frames)


    current_frames = 0
    for set_num in target_frames:
        target_path = os.path.join(OUTPUT_DIR , set_num)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for v_num, file_list in target_frames[set_num].items():
            current_frames += detection_to_file(target_path, v_num, file_list, detect, total_frames, current_frames)


def render_image(img_path, dets, inds, target_dir, threshold=85):
    im = cv2.imread(img_path)[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1] * 100


        #Fix bug 6
        x1 = bbox[0]
        y1 = bbox[1] 
        width = bbox[2] 
        height =  bbox[3]
        if score >  threshold:
            rectangle = plt.Rectangle((x1,y1), width, height, fill=False, edgecolor="green", linewidth=3.5)
            ax.add_patch(rectangle)
            ax.text(x1, y1 - 2, "{}%".format(score),
                    bbox=dict(facecolor="green", alpha=0.5),
                    fontsize=14, color='white')


    
    
        
    image_name = os.path.basename(img_path)
    ax.set_title(image_name)  
    plt.axis('off')
    plt.tight_layout()
    target_path = os.path.join(target_dir, "dt_"+image_name)

    print("Image save to {}".format(target_path))
    plt.savefig(target_path)
            
            
            
if __name__ == "__main__":
    args = parse_args()
    
    
    
    assert(args.data in data_info, "Dataset '{}' does not exist in data_info.py.".format(args.data))
    data = data_info[args.data]
    
    
    #DATA_PATH = "/root/data/caltech-pedestrian-dataset-converter/data/"
    #IMG_PATH = os.path.join(DATA_PATH + "images")
    
    global GPU_ID
    global OUTPUT_DIR
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    GPU_ID = args.gpu_id
    

    OUTPUT_DIR= os.path.join(data["data_path"],"res" ,  args.output)
    if not os.path.exists(OUTPUT_DIR ):
        os.makedirs(OUTPUT_DIR ) 
    
    
    
    
    print("Loading Network")
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
 
    print("MC-CNN model loaded")
    print(data)
    print("Start Detecting")
    write_testing_results_file(net, data)                 