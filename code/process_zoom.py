import os
import pandas as pd
from args import process_parser
from joblib import Parallel, delayed
import SimpleITK as sitk
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from imgtools.ops import Resize

def find_bbox(mask: sitk.Image) -> np.ndarray:
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)
    xstart, ystart, zstart, xsize, ysize, zsize = stats.GetBoundingBox(1)
    
    # Prevent the following ITK Error from SmoothingRecursiveGaussianImageFilter: 
    # The number of pixels along dimension 2 is less than 4. This filter requires a minimum of four pixels along the dimension to be processed.
    if xsize < 4:
        xsize = 4
    if ysize < 4:
        ysize = 4
    if zsize < 4:
        zsize = 4

    xend, yend, zend = xstart + xsize, ystart + ysize, zstart + zsize
    return xstart, xend, ystart, yend, zstart, zend

def crop_bbox(image: sitk.Image, bbox_coords, input_size) -> sitk.Image:
    min_x, max_x, min_y, max_y, min_z, max_z = bbox_coords
    img_x, img_y, img_z = image.GetSize()

    if min_x < 0: 
        min_x, max_x = 0, input_size[0]
    elif max_x > img_x: # input_size[0]:
        min_x, max_x = img_x - input_size[0], img_x

    if min_y < 0:
        min_y, max_y = 0, input_size[1]
    elif max_y > img_y: # input_size[1]:
        min_y, max_y = img_y - input_size[1], img_y

    if min_z < 0:
        min_z, max_z = 0, input_size[2]
    elif max_z > img_z: # input_size[2]:
        min_z, max_z = img_z - input_size[2], img_z
    
    img_crop = image[min_x:max_x, min_y:max_y, min_z:max_z]
    img_crop = Resize(input_size)(img_crop)
    return img_crop

def get_row(row, input_path, output_path):
    ct_path = os.path.join(input_path, row.output_folder_CT, "CT.nii.gz")
    mask_path = os.path.join(input_path, row.output_folder_RTSTRUCT_CT, "GTVp.nii.gz")
    if os.path.exists(ct_path) and os.path.exists(mask_path):
        # get GTVp centroid
        img  = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        
        assert img.GetSize() == mask.GetSize()

        bbox     = find_bbox(mask)
        img_crop = crop_bbox(img, bbox, (50, 50, 50))

        # save crop and row
        crop_path = os.path.join(output_path, f"{row.code}.nii.gz")
        sitk.WriteImage(img_crop, crop_path)
        return crop_path, 0, 0, 0
    
    return
    
def main():
    params = process_parser().parse_args()
    # reads a Med-ImageTools processed dataset and outputs a csv file of:
    #   - `image_path`: path of image
    #   - `coordX, coordY, coordZ`: coordinates of the center of GTVp

    df_data = pd.read_csv(os.path.join(params.input_path, "dataset.csv"), index_col=2).dropna(axis=0, subset=["output_folder_CT"])
    # multi process
    rows = Parallel(n_jobs=-1)(delayed(get_row)(row, params.input_path, params.output_path) for row in tqdm(df_data.itertuples(), total=len(df_data)))

    # convert to dataframe
    df_new  = pd.DataFrame(columns=["image_path", "coordX", "coordY", "coordZ"])
    for row in rows:
        if row is not None:
            df_new = pd.concat([df_new, pd.DataFrame([row], columns=["image_path", "coordX", "coordY", "coordZ"])], ignore_index=True)
    df_new.to_csv(params.save_path, index=False)
    
if __name__ == '__main__':
    main()