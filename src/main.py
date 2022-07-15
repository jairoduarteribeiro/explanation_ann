import pandas as pd
import tensorflow as tf
import numpy as np
import logging

from keras.models import load_model
from datasets.dataset_utils import get_dataset_path
from explanation.explanation_utils import get_minimal_explanation, log_explanation
from models.model_utils import get_model_path
from explanation.milp import build_network


logging.basicConfig(
    filename='explanation_box.log',
    filemode='w',
    format=f'{"".ljust(120, "-")}\n%(asctime)s\n\n%(message)s')
logger = logging.getLogger('explanation')
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    train_path = get_dataset_path('iris', 'train.csv')
    train_data = pd.read_csv(train_path)
    test_path = get_dataset_path('iris', 'test.csv')
    test_data = pd.read_csv(test_path)
    features = train_data.columns[:-1]
    dataframe = pd.concat([train_data, test_data], ignore_index=True)
    model_path = get_model_path('iris.h5')
    model = load_model(model_path)
    layers = model.layers
    mdl, bounds = build_network(model, dataframe, 'tjeng')
    for data_idx, data in test_data.iterrows():
        logger.info(f'Getting explanation for data {data_idx}\n{data}')
        network_input = tf.reshape(data.iloc[:-1], (1, -1))
        network_output = np.argmax(model.predict(network_input))
        logger.info(f'Predicted output: {network_output}')
        mdl_clone = mdl.clone()
        explanation = get_minimal_explanation(mdl_clone, bounds, 'tjeng', network_input, network_output, layers, True)
        log_explanation(logger, features, explanation)
