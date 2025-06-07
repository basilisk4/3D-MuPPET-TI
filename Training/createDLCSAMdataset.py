import os
import shutil
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import argparse
import distutils.dir_util
from tqdm import tqdm
import yaml
import numpy as np


    
def crop(image, threshold=40):
    image_height, image_width = image.size
    # Calculate center of each side
    center_x = image_width // 2
    center_y = image_height // 2
    # Initialize offset
    l_off = r_off = t_off = b_off = -1
    i = 0

    def pixel_sum(x, y):
        pixel = image.getpixel((x, y))
        return sum(pixel) if isinstance(pixel, tuple) else pixel
    
    # Find Offset for each side
    while i < min(image_width, image_height) and (l_off == -1 or r_off == -1 or t_off == -1 or b_off == -1):
        if l_off == -1 and pixel_sum(i, center_y) > threshold:
            l_off = i
        if r_off == -1 and pixel_sum(image_width - i - 1, center_y) > threshold:
            r_off = image_width - i - 1
        if t_off == -1 and pixel_sum(center_x, i) > threshold:
            t_off = i
        if b_off == -1 and pixel_sum(center_x, image_height - i - 1) > threshold:
            b_off = image_height - i - 1
        i += 1

    # Crop the image based on detected offsets
    border=[l_off, t_off,r_off,b_off]
    crop_image = image.crop(border)
    return crop_image, border

def segment(image, predictor):
    # Get the image dimensions
    height, width, _ = image.shape
    #hand image to sam
    predictor.set_image(image)
    input_box = np.array([0,0,width,height])
    # Perform the segmentation
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,  # Only return the most confident mask
    )
    # Convert the mask to uint8 format (0 for black, 255 for white)
    mask_bw = ((1-masks[0]) * 255).astype(np.uint8)
    return Image.fromarray(mask_bw)
        


def createTrainingData(dlcDir, outDir, ckpt, type):
    if os.path.exists(outDir):
        if len(os.listdir(outDir))>0:
            raise Exception(outDir+" is not an empty directory.") 
    else :
        os.makedirs(outDir)
        
    dataDir= "labeled-data"
    os.makedirs(os.path.join(outDir,dataDir))
      
    dlcDataset= os.path.join(dlcDir, "training-datasets")
    distutils.dir_util.copy_tree(dlcDataset, os.path.join(outDir, "training-datasets"))
    

    # Initialize the model
    sam = sam_model_registry[type](checkpoint=ckpt)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    # Set up the predictor
    predictor = SamPredictor(sam)
    
    labelDir="labeled-data"
    for dir in os.listdir(os.path.join(dlcDir,labelDir)):
        dataDir=os.path.join(dlcDir,labelDir,dir)
        saveDir=os.path.join(outDir,labelDir,dir)
        os.makedirs(os.path.join(outDir,labelDir,dir))
        print("Process "+dir)
        for file in tqdm(os.listdir(dataDir)):
            if file.endswith(".jpg"):
                # Load and remove padding the image
                image = Image.open(os.path.join(dataDir, file)).convert("RGB")
                crop_img, padding=crop(image)
                #segment image
                mask=segment(crop_img, predictor)
                #add padding to mask to align with original image
                result = Image.new("1", image.size)
                result.paste(mask, (padding[0], padding[1]))
                #save image
                result.save(os.path.join(saveDir,file))
            else:
                shutil.copy(os.path.join(dataDir,file), os.path.join(saveDir,file))
                
                
      
    config_path=os.path.join(dlcDir,"config.yaml")      
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the project_path field
    config['project_path'] = os.path.abspath(outDir)

    # Write back to the file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlc", type=str, default="./TrainingData/N6000/DLC", help="Path to DLC training dataset")
    parser.add_argument("--out", type=str, default="./TrainingData/DLCSAM", help="Path to save the created DLC training dataset")
    parser.add_argument("--SamCkpt", type=str, default="./Weights/sam_vit_h_4b8939.pth", help="Path to model checkpoint")
    parser.add_argument("--SamType", type=str, default="vit_h", help="Checkpoint size provided to SamCkpt")

    args = parser.parse_args()
    createTrainingData(args.dlc, args.out, args.ckpt, args.type)