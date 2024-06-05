# check array sizes of all crops in args.input_path

from argparse import ArgumentParser
import os
import pandas as pd
import SimpleITK as sitk

def main():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=str, help='path to mapping csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    print(df.shape)
    for n, path in enumerate(df.image_path.tolist()):
        # check if all crops are the same size
        crop = sitk.ReadImage(path)
        print(n, crop.GetSize())

main()