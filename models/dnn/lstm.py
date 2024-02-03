from tensorflow.keras.layers import LSTM as KERAS_LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from .dnn import DNN

class LSTM(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(LSTM, self).__init__(model, trained)

    @classmethod
    def make(
        cls,
        input_shape: int,
        rnn_size: int,
        hidden_size: int,
        dropout: float = 0.5,
        n_classes: int = 6,
        lr: float = 0.001
    ):
        model = Sequential()

        model.add(KERAS_LSTM(rnn_size, input_shape=(1, input_shape)))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_size, activation='relu'))

        model.add(Dense(n_classes, activation='softmax'))
        optimzer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])

        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        return data
