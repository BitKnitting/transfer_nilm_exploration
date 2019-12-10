
import pandas as pd
import numpy as np
import python_lib.data_generator
import python_lib.ziptodf
from keras import backend as K


# filename = 'created_data/REDD/microwave/microwave_training_.pkl.zip'
# df = pd.read_pickle(filename)
a = np.arange(20)
a = a.reshape((10,2))
df = pd.DataFrame(a)
df.columns = ['aggregate','microwave']

python_lib.ziptodf.print_stats(df, '\n\nMicrowave Training Data')
tra_provider = python_lib.data_generator.DataGenerator(3, 5, True)
for batch in tra_provider.get_batch(df):
    x_train, y_train = batch
    x_train = K.cast_to_floatx(x_train)
    y_train = K.cast_to_floatx(y_train)
    print(x_train[0:])
