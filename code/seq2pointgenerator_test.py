from python_lib.Seq2PointGenerator import Seq2PointGenerator
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape


a = np.arange(20)
a = a.reshape((10, 2))
df = pd.DataFrame(a)
df.columns = ['aggregate', 'microwave']
batchsize = 5
windowsize = 3
gen = Seq2PointGenerator(df, windowsize=windowsize, batchsize=batchsize)
model = Sequential()
# The inp numpy array has batchsize rows.  Each row has window size number of readings.

# model.add(Reshape((-1, windowsize, 1), input_shape=(1, windowsize)))
model.add(Dense(1, activation='relu', input_shape=(windowsize,)))
model.summary()
model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=gen)
