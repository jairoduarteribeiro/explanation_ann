import pandas as pd
from tensorflow.keras.models import load_model
from datasets.dataset_utils import get_dataset_path
from models.model_utils import get_model_path
from milp import build_network


def main():
    train_path = get_dataset_path('heart_disease', 'train.csv')
    train_data = pd.read_csv(train_path)
    test_path = get_dataset_path('heart_disease', 'train.csv')
    test_data = pd.read_csv(test_path)
    dataframe = pd.concat([train_data, test_data], ignore_index=True)
    model_path = get_model_path('heart.h5')
    model = load_model(model_path)
    mdl, output_bounds = build_network(model, dataframe, 'tjeng')


if __name__ == '__main__':
    main()
