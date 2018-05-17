from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD, Nadam

def get_lstm(xshape, yshape, BATCH_SIZE):

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, batch_input_shape=(BATCH_SIZE, xshape[1], xshape[2])))
    model.add(LSTM(32))
    model.add(Dense(2, activation='sigmoid'))
    opt = Nadam(lr=0.00005 )
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model,opt
