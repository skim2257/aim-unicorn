import os
import pandas as pd
from args import process_parser
from joblib import Parallel, delayed
import SimpleITK as sitk
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

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


def get_row(row, input_path):
    ct_path = os.path.join(input_path, row.output_folder_CT, "CT.nii.gz")
    mask_path = os.path.join(input_path, row.output_folder_RTSTRUCT_CT, "GTVp.nii.gz")
    if os.path.exists(ct_path) and os.path.exists(mask_path):
        # get GTVp centroid
        mask = sitk.ReadImage(mask_path)
        centroid = find_centroid(mask)
        

        # return row
        return ct_path, centroid[0], centroid[1], centroid[2]
    return
    

def main():
    params = process_parser().parse_args()
    # reads a Med-ImageTools processed dataset and outputs a csv file of:
    #   - `image_path`: path of image
    #   - `coordX, coordY, coordZ`: coordinates of the center of GTVp

    df_data = pd.read_csv(os.path.join(params.input_path, "dataset.csv"), index_col=2).dropna(axis=0, subset=["output_folder_CT"])
    # multi process
    rows = Parallel(n_jobs=-1)(delayed(get_row)(row, params.input_path) for row in tqdm(df_data.itertuples(), total=len(df_data)))

    # convert to dataframe
    df_new  = pd.DataFrame(columns=["image_path", "coordX", "coordY", "coordZ"])
    for row in rows:
        if row is not None:
            df_new = pd.concat([df_new, pd.DataFrame([row], columns=["image_path", "coordX", "coordY", "coordZ"])], ignore_index=True)
    df_new.to_csv(os.path.join(params.save_path), index=False)
            


if __name__ == '__main__':
    main()
