from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD, Nadam

def get_mlp(xshape, yshape, BATCH_SIZE):

    model = Sequential()
    model.add(Dense(2, activation='sigmoid', batch_input_shape=(BATCH_SIZE, xshape[1])))
    #model.add(Dense(500, activation='sigmoid'))
    #model.add(Dense(2, activation='sigmoid'))
    opt = Nadam(lr=0.000001 )
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model,opt
