from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.activations import relu, softmax
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from time import time
from models.model_utils import get_model_path
from datasets.iris import iris

# Load Iris dataset
(x_train, y_train), (x_test, y_test) = iris.load_data()

# Parameters
n_classes = 3
n_hidden_layers = 4
n_neurons = 16
n_epochs = 100
batch_size = 4

# Create model
model = Sequential([InputLayer(input_shape=(x_train.shape[1]))])
for _ in range(n_hidden_layers - 1):
    model.add(Dense(n_neurons, activation=relu))
model.add(Dense(n_classes, activation=softmax))
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])

# Callbacks
early_stopping = EarlyStopping(patience=20)
model_checkpoint = ModelCheckpoint(filepath=get_model_path('iris.h5'), save_best_only=True)

# Train and save model
if __name__ == '__main__':
    start_time = time()
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint])
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')
