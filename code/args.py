from argparse import ArgumentParser
import os
import dotenv

dotenv.load_dotenv()

def parser():
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, default=os.environ["DATA_PATH"], help='Path to the CSV with the data')
    parser.add_argument('--save_path', type=str, default="data/fmcib.csv", help='Path to the output directory')  
    parser.add_argument('--weights_path', type=str, default=os.environ["MODEL_PATH"], help='Path to the model weights')
    parser.add_argument('--precropped', action='store_true', help='Whether to use the precropped data')
    parser.parse_args()

    return parser

def process_parser():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to the input directory')
    parser.add_argument('--output_path', type=str, default=os.environ["ZOOM_PATH"], help='Path to the output directory')
    parser.add_argument('--save_path', type=str, default=os.environ["DATA_PATH"], help='Path to the output file')
    return parser

