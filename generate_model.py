from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.activations import relu, softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_utils import model_path
from time import time
from datasets.heart_disease import heart

# Load Glass dataset and normalize it
(x_train, y_train), (x_test, y_test) = heart.load_data()
x_train = normalize(x_train)
x_test = normalize(x_test)

# Parameters
n_classes = 2
n_hidden_layers = 4
n_neurons = 64
n_epochs = 1000
batch_size = 4

# Create model
model = Sequential([Input(shape=(x_train.shape[1],))])
for _ in range(n_hidden_layers):
    model.add(Dense(n_neurons, activation=relu))
model.add(Dense(n_classes, activation=softmax))
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])

# Callbacks
early_stopping = EarlyStopping(patience=20)
model_checkpoint = ModelCheckpoint(filepath=model_path('heart'),
                                   save_best_only=True)

if __name__ == '__main__':
    # Train and save model
    start_time = time()
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint])
    end_time = time()
    print(f'Time of training: {end_time - start_time:.2f} seconds.')
