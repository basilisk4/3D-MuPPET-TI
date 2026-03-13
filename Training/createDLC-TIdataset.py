import json
import os
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
from PIL import Image
import csv
from tqdm import tqdm
import math
import shutil
import deeplabcut
import argparse



def read_json_file(file_path):
    try:
        # Open and load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

def segment(inputFolder, outputFolder, samPredictor, dataset):
    annotations=dataset["Annotations"]
    keypoints=[]
    i=0
    for annot in tqdm(annotations):
        InferImage=cv2.imread(os.path.join(inputFolder, annot["Path"]))
        InferImage= cv2.cvtColor(InferImage, cv2.COLOR_BGR2RGB)
        for ind in annot["BirdID"]:
            bbox=annot["BBox"][ind]
            
            center=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]
            #compute top left corner
            tl=[round(center[0])-512,round(center[1])-512]
            if tl[0] < 0: tl[0]=0
            if tl[1] < 0: tl[1]=0
            if tl[0] > 2816: tl[0]=2816
            if tl[1] > 1136: tl[1]=1136
            
            Crop = InferImage[tl[1]:tl[1]+1023,tl[0]:tl[0]+1023]
            inputBox=[bbox[0]-tl[0],bbox[1]-tl[1],bbox[2]-tl[0],bbox[3]-tl[1]]
            
            samPredictor.set_image(Crop)
            masks, scores, logits = samPredictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(inputBox),
                multimask_output=False,  # Only return the most confident mask
            )
            mask_bw = ((masks[0]) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            box=[1023,1023,0,0]
            for contour in contours:
            # Get the tightest bounding box around the segmented object
                x, y, w, h = cv2.boundingRect(contour) 
                #check if boundingbox is completly outside of the Yolo/Groundtruth bounding box
                if x<inputBox[0] and  x+w<inputBox[0]: continue
                if x>inputBox[2] and  x+w>inputBox[2]: continue
                if y<inputBox[1] and  y+h<inputBox[1]: continue
                if y>inputBox[3] and  y+h>inputBox[3]: continue
                box=[min(box[0],x), min(box[1],y), max(box[2], x+w), max(box[3], y+h)]
            cropped_mask = mask_bw[box[1]:box[3], box[0]:box[2]]
            path=os.path.join(outputFolder, str(i)+".png")
            Image.fromarray(cropped_mask).save(path)
            total_bbox=np.add(box, tl+tl)
            entry=[os.path.join("labeled-data",path.split("/labeled-data/")[1])]
            for key in dataset["info"]["Keypoints"]:
                point=annot["Keypoint2D"][ind][key]
                entry.append(point[0]-total_bbox[0])
                entry.append(point[1]-total_bbox[1])
            #print(entry)
            keypoints.append(entry)
            i=i+1       
    
    return keypoints   

def resize(input, inputFolder, outputFolder, outputCSV):
    keypoints=[['scorer','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex','Alex'],
               ['bodyparts','hd_beak','hd_beak','hd_leftEye','hd_leftEye','hd_rightEye','hd_rightEye','hd_nose','hd_nose','bp_leftShoulder','bp_leftShoulder','bp_rightShoulder','bp_rightShoulder','bp_topKeel','bp_topKeel','bp_bottomKeel','bp_bottomKeel','bp_tail','bp_tail'],
               ['coords','x','y','x','y','x','y','x','y','x','y','x','y','x','y','x','y','x','y']]
    filtered=0
    corrected=0
    for line in  tqdm(input):
        if not line[0].endswith('.png'): continue
        image_path=line[0] 
        image=cv2.imread(os.path.join(inputFolder, image_path))
        height, width = image.shape[:2]
        sc_factor=320/max(width,height)
        resized_image = cv2.resize(image, None, fx=sc_factor, fy=sc_factor,interpolation=cv2.INTER_NEAREST)
        resized_height, resized_width=resized_image.shape[:2]
        x_padding=(320-resized_width)/2
        y_padding=(320-resized_height)/2
        if x_padding<0 or y_padding<0 :print(x_padding,y_padding)
        padded_image = cv2.copyMakeBorder(resized_image, math.floor(y_padding), math.ceil(y_padding), math.floor(x_padding), math.ceil(x_padding), cv2.BORDER_CONSTANT,value=(0, 0, 0))
        cv2.imwrite(os.path.join(outputFolder,image_path),padded_image)
        entry=[image_path]
        err=10
        outlier=False
        i=1
        while i <19 and not outlier:
            corr=False
            if float(line[i])<  -err or float(line[i])> width+err:
                outlier=True
                continue
            if float(line[i])<0 :
                line[i]=0
                corr=True
            if float(line[i])>= width :
                line[i]=width-1
                corr=True
            point_x=float(line[i])*sc_factor+x_padding
            entry.append(point_x)
            i=i+1
            if float(line[i])<  -err or float(line[i])> height+err:
                outlier=True
                continue
            if float(line[i])<0 :
                line[i]=0
                corr=True
            if float(line[i])>= height :
                line[i]=height-1
                corr=True
            point_y=float(line[i])*sc_factor+y_padding
            entry.append(point_y)
            i=i+1
            if corr: 
                corrected += 1
        if outlier:
            filtered += 1
            continue
        #print(entry)
        keypoints.append(entry)
    print("corrected: "+str(corrected)+"  filtered: "+ str(filtered))
    with open(outputCSV, 'w') as file:
        csv.writer(file).writerows(keypoints)    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlc", type=str, default="./TrainingData/N6000/DLC", help="Path to DLC training dataset")
    parser.add_argument("--out", type=str, default="./TrainingData/DLC-TI", help="Path to save the created DLC training dataset")
    parser.add_argument("--SamCkpt", type=str, default="./Weights/sam_vit_h_4b8939.pth", help="Path to model checkpoint")
    parser.add_argument("--SamType", type=str, default="vit_h", help="Checkpoint size provided to SamCkpt")

    args = parser.parse_args()

    sam_checkpoint = args.SamCkpt  # Update with your checkpoint path
    model_type = args.SamType  # Model type: vit_b, vit_l, or vit_h
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    samPredictor = SamPredictor(sam)

    inputFolder= args.dlc
    DLC_folder= args.out   

    dataset=read_json_file(os.path.join(inputFolder,"Annotation/Train-2D.json"))
    outputFolder=os.path.join(DLC_folder,"labeled-data/Video_Train")
    os.makedirs(outputFolder)
    csvPath=os.path.join(outputFolder,"CollectedData_Alex.csv")
    keypoints=segment(inputFolder, outputFolder, samPredictor, dataset)
    resize(keypoints,DLC_folder,DLC_folder,csvPath)

    dataset=read_json_file(os.path.join(inputFolder,"Annotation/Val-2D.json"))
    outputFolder=os.path.join(DLC_folder,"labeled-data/Video_Val")
    os.makedirs(outputFolder)
    csvPath=os.path.join(outputFolder,"CollectedData_Alex.csv")
    keypoints=segment(inputFolder, outputFolder, samPredictor, dataset)
    resize(keypoints,DLC_folder,DLC_folder,csvPath)

    # copy config
    src = os.path.join(inputFolder,"DLC/config.yaml")
    dst = os.path.join(DLC_folder,"config.yaml")
    shutil.copy(src, dst)

    deeplabcut.convertcsv2h5(dst, scorer= 'Alex')



