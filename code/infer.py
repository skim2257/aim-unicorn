from fmcib.run import get_features
from args import parser
import os

if __name__ == "__main__":
    params = parser().parse_args()

    dir_path = os.path.dirname(params.save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    feature_df = get_features(params.input_path, 
                              weights_path=params.weights_path, 
                              precropped=params.precropped)
    feature_df.drop(columns=['coordX', 'coordY', 'coordZ'], inplace=True)
    
    feature_df.to_csv(params.save_path, index=False)
    print(feature_df)