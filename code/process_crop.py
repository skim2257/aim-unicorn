import os
import pandas as pd
from args import process_parser
from joblib import Parallel, delayed
import SimpleITK as sitk
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from imgtools.ops import Resize

def find_centroid(mask: sitk.Image) -> np.ndarray:
    """Find the centroid of a binary image in image
    coordinates.

    Parameters
    ----------
    mask
        The bimary mask image.

    Returns
    -------
    np.ndarray
        The (x, y, z) coordinates of the centroid
        in image space.
    """
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)
    return np.asarray(centroid_idx, dtype=np.float32)

def crop_centroid(image: sitk.Image, 
                  centroid: tuple, 
                  input_size: tuple) -> sitk.Image:
    blank = np.zeros(input_size)

    min_x = int(centroid[0] - input_size[0] // 2)
    max_x = int(centroid[0] + input_size[0] // 2)
    min_y = int(centroid[1] - input_size[1] // 2)
    max_y = int(centroid[1] + input_size[1] // 2)
    
    # tuning where in the neck to crop
    min_z = int(centroid[2] - input_size[2] // 2)
    max_z = int(centroid[2] + input_size[2] // 2)

    img_x, img_y, img_z = image.GetSize()

    if min_x < 0: 
        min_x, max_x = 0, input_size[0]
    elif max_x > img_x: 
        min_x, max_x = img_x - input_size[0], img_x

    if min_y < 0:
        min_y, max_y = 0, input_size[1]
    elif max_y > img_y: 
        min_y, max_y = img_y - input_size[1], img_y

    if min_z < 0:
        min_z, max_z = 0, input_size[2]
    elif max_z > img_z: 
        min_z, max_z = img_z - input_size[2], img_z
    
    return image[min_x:max_x, min_y:max_y, min_z:max_z]

def get_row(row, input_path, output_path):
    ct_path = os.path.join(input_path, row.output_folder_CT, "CT.nii.gz")
    mask_path = os.path.join(input_path, row.output_folder_RTSTRUCT_CT, "GTVp.nii.gz")
    if os.path.exists(ct_path) and os.path.exists(mask_path):
        # get GTVp centroid
        img  = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        
        assert img.GetSize() == mask.GetSize()

        centroid = find_centroid(mask)
        img_crop = crop_centroid(img, centroid, (50, 50, 50))

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