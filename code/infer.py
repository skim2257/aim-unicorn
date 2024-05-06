from fmcib.run import get_features
from args import parser

if __name__ == "__main__":
    params = parser().parse_args()

    feature_df = get_features(params.input_path, 
                              weights_path=params.weights_path, 
                              precropped=params.precropped)
    feature_df.drop(columns=['coordX', 'coordY', 'coordZ'], inplace=True)
    
    feature_df.to_csv(params.save_path, index=False)
    print(feature_df)