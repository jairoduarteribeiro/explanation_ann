import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from datasets.dataset_utils import get_dataset_path
from models.model_utils import get_model_path
from milp import build_network
from explanation import get_minimal_explanation


def main():
    train_path = get_dataset_path('heart_disease', 'train.csv')
    train_data = pd.read_csv(train_path)
    test_path = get_dataset_path('heart_disease', 'test.csv')
    test_data = pd.read_csv(test_path)
    dataframe = pd.concat([train_data, test_data], ignore_index=True)
    model_path = get_model_path('heart.h5')
    model = load_model(model_path)
    mdl, output_bounds = build_network(model, dataframe, 'tjeng')
    for data_idx, data in test_data.iterrows():
        network_input = tf.reshape(data.iloc[:-1], (1, -1))
        network_output = np.argmax(model.predict(network_input))
        mdl_clone = mdl.clone()
        explanation = get_minimal_explanation(mdl_clone, output_bounds, 'tjeng', network_input, network_output, 2)
        print(f'Explanation for data {data_idx}: {explanation}')


if __name__ == '__main__':
    main()
