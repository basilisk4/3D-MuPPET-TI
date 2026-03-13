""" inference on single video for YOLO + DLC"""
import cv2 
from ultralytics import YOLO
import torch
import argparse
import numpy as np
import math

import sys
sys.path.append("./Repositories/DeepLabCut-live")

import deeplabcut as dlc
from dlclive import DLCLive, Processor
from segment_anything import SamPredictor, sam_model_registry






def DLCInference(InferFrame,inBox,dlc_liveObj,samPredictor):
    """Inference for DLC"""
    InferFrame=cv2.cvtColor(InferFrame, cv2.COLOR_BGR2RGB)
    inBox = [0 if val < 0 else val for val in inBox] #f out of screen, 0
    #create 1024x1024 crop window around bounding box and segment with SAM
    center=[(inBox[0]+inBox[2])/2,(inBox[1]+inBox[3])/2]
    #compute top left corner
    tl=[round(center[0])-512,round(center[1])-512]
    if tl[0] < 0: tl[0]=0
    if tl[1] < 0: tl[1]=0
    if tl[0] > 2816: tl[0]=2816
    if tl[1] > 1136: tl[1]=1136
    
    inBox =[inBox[0]-tl[0],inBox[1]-tl[1],inBox[2]-tl[0],inBox[3]-tl[1]]
    
    Crop = InferFrame[tl[1]:tl[1]+1023,tl[0]:tl[0]+1023]
    samPredictor.set_image(Crop)
    masks, scores, logits = samPredictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(inBox),
        multimask_output=False,  # Only return the most confident mask
    )
    mask_bw = ((masks[0]) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outBox=[1023,1023,0,0]
    for contour in contours:
    # Get the tightest bounding box around the segmented object
        x, y, w, h = cv2.boundingRect(contour) 
        #check if boundingbox is completly outside of the Yolo/Groundtruth bounding box
        if x<inBox[0] and  x+w<inBox[0]: continue
        if x>inBox[2] and  x+w>inBox[2]: continue
        if y<inBox[1] and  y+h<inBox[1]: continue
        if y>inBox[3] and  y+h>inBox[3]: continue
        outBox=[min(outBox[0],x), min(outBox[1],y), max(outBox[2], x+w), max(outBox[3], y+h)]
    cropped_mask = mask_bw[outBox[1]:outBox[3], outBox[0]:outBox[2]]
    height, width = cropped_mask.shape[:2]
    sc_factor=320/max(width,height)
    resized_image = cv2.resize(cropped_mask, None, fx=sc_factor, fy=sc_factor,interpolation=cv2.INTER_NEAREST)
    resized_height, resized_width=resized_image.shape[:2]
    x_padding=(320-resized_width)/2
    y_padding=(320-resized_height)/2
    padded_image = cv2.copyMakeBorder(resized_image, math.floor(y_padding), math.ceil(y_padding), math.floor(x_padding), math.ceil(x_padding), cv2.BORDER_CONSTANT,value=(0, 0, 0))
    #Image.fromarray(padded_image).show()

    if dlc_liveObj.sess == None: #if first time, init
        DLCPredict2D = dlc_liveObj.init_inference(padded_image)

    DLCPredict2D= dlc_liveObj.get_pose(padded_image)    
    # remove padding
    DLCPredict2DList = [[DLCPredict2D[j,0]-math.floor(x_padding),DLCPredict2D[j,1]-math.floor(y_padding)] for j in range(DLCPredict2D.shape[0])]
    # revert scaling
    DLCPredict2DList = [[j[0]*(1/sc_factor),j[1]*(1/sc_factor)] for j in DLCPredict2DList]
    # convert to global coordinates
    outBox =[outBox[0]+tl[0],outBox[1]+tl[1],outBox[2]+tl[0],outBox[3]+tl[1],]
    DLCPredict2DList = [[j[0]+outBox[0],j[1]+outBox[1]] for j in DLCPredict2DList]
    print(DLCPredict2DList)
    DLCPredict2DList=[DLCPredict2DList[4],DLCPredict2DList[5],DLCPredict2DList[6],
                      DLCPredict2DList[7],DLCPredict2DList[8],DLCPredict2DList[0],
                      DLCPredict2DList[3],DLCPredict2DList[1],DLCPredict2DList[2]]
    return DLCPredict2DList


def VisualizeAll(frame, box, DLCPredict2D):
    """Visualize all stuff"""
    colourList = [(255,255,0),(255,0 ,255),(128,0,128),(203,192,255),(0, 255, 255),(255, 0 , 0 ),(63,133,205),(0,255,0),(0,0,255)]
    ##Order: Lshoulder, Rshoulder, topKeel,botKeel,Tail,Beak,Nose,Leye,Reye
    ##Points:
    # PlotPoints = []
    for x,point in enumerate(DLCPredict2D):
        roundPoint = [round(point[0]+box[0]),round(point[1]+box[1])]
        cv2.circle(frame,roundPoint,1,colourList[x], 3)

    cv2.rectangle(frame,(round(box[0]),round(box[1])),(round(box[2]),round(box[3])),[255,0,0],3)

    return frame


def RunInference(YOLOPath,DLCWeight, SamType, SamCkpt, InputVideo,CropSize,startFrame=0,ScaleBBox=1):
    
    YOLOModel = YOLO(YOLOPath)

    dlc_proc = Processor()
    dlc_liveObj = DLCLive(DLCWeight, processor=dlc_proc)
    
    # Initialize the model
    sam = sam_model_registry[SamType](checkpoint=SamCkpt)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    # Set up the predictor
    predictor = SamPredictor(sam)
    
    cap = cv2.VideoCapture(InputVideo)
    cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
    imsize = (int(cap.get(3)),int(cap.get(4)))
    counter=startFrame

    cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    TotalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(filename="YOLODLC2D_sample.mp4", apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    # while(cap.isOpened()):
    for i in range(1800):

        ret, frame = cap.read()
        # print(counter)

        if ret == True:
            InferFrame = frame.copy()
            InferFrame = InferFrame
            # InferFrame = torch.tensor(InferFrame).to("cuda")
            results = YOLOModel(InferFrame, imgsz=3840,device="cpu")
            # results = YOLOModel(InferFrame,device="cpu")

            ##Filter for birds:
            classID = [key for key,val in results[0].names.items() if val == "bird"][0]
            # frame = results[0].plot()
            DetectedClasses = results[0].boxes.cls.cpu().numpy().tolist()
            
            # bbox = results[0].boxes.xyxy.cpu().numpy().tolist()
            bbox = results[0].boxes.xywh.cpu().numpy().tolist()
            ##Filter birds only:
            bbox = [box for x,box in enumerate(bbox) if DetectedClasses[x] == classID]


            bbox = [[box[0],box[1],box[2]*ScaleBBox,box[3]*ScaleBBox] for box in bbox] #scale width and height
            ##convert back to xyxy:
            bboxXY = [[box[0]-(box[2]/2), box[1]-(box[3]/2),box[0]+(box[2]/2),box[1]+(box[3]/2)] for box in bbox]

            # import ipdb;ipdb.set_trace()
            for box in bboxXY:
                DLCPredict2D= DLCInference(InferFrame,box,dlc_liveObj, predictor)
                frame = VisualizeAll(frame, box, DLCPredict2D)

            out.write(frame)

            cv2.imshow('Frame',frame)
            # import ipdb;ipdb.set_trace()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()
    out.release()

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Input Video, path to input video")
    parser.add_argument("--YOLOweight",
                        type=str,
                        default= "Weights/YOLO_Barn.pt",
                        help="Path to pre-trained weight for YOLO model")
    parser.add_argument("--DLCweight",
                        type=str,
                        default= "Weights/DLC_Barn/",
                        help="Path to pre-trained weight for exported DLC model directory")
    parser.add_argument("--SamType",
                        type=str,
                        default= "vit_h",
                        help="Specify the Sam model type used.")
    parser.add_argument("--SamCkpt",
                        type=str,
                        default= "Weights/sam_vit_h_4b8939.pth",
                        help="Path to sam weight dict")

    arg = parser.parse_args()

    return arg



if __name__ == "__main__":

    args = ParseArgs()
    CropSize = (320,320)
    
    RunInference(args.YOLOweight,args.DLCweight,args.SamType,args.SamCkpt, args.input,CropSize,startFrame=0,ScaleBBox=1)

