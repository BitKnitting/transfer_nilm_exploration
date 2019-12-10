import numpy as np
import keras
from keras import backend as K
#
# The goal of this code is to create a Keras Sequence class that will work keras's
# model.fit_generator() method.
#
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#
# See: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# For documentation on implementing the Sequence() abstract base class.


class Seq2PointSequence(keras.utils.Sequence):
    def __init__(self, df, windowsize=599, batchsize=1000,  shuffle=True):
        self.windowsize = windowsize
        self.shuffle = shuffle
        self.batchsize = batchsize
        np_array = np.array(df)
        self.inputs, self.targets = np_array[:, 0], np_array[:, 1]
        self.inputs = K.cast_to_floatx(self.inputs)
        self.targets = K.cast_to_floatx(self.targets)

    def __len__(self):
        return int(np.ceil(len(self.inputs)/self.batchsize))

    def __getitem__(self, index):
        # excerpt gets us batchsize rows.  Each row has windowsize number of readings.  The readings are in order
        # The researchers note "sliding window" here where I think of it more as a random place to start grabbing
        # a windowsize bunch of readings.
        for start_idx in range(0, self.max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            # The input is a 2D array where number of rows = batchsize and number of cols = windowsize
            inp = np.array([inputs[idx:idx + self.windowsize]
                            for idx in excerpt])
            # ---------------------
            # From the paper: "...the output is the midpoint element x of the corresponding
            # window of the target appliance, where T = t + W/2...The intuition behind
            # this assumption is that we expect the state of the midpoint element of
            # that appliance to relate to the information of mains before and after
            # that midpoint."
            # -----------------------
            # The shape of the target/output is (batchsize,1) where each row is of shape (1,)
            # The entry is the midpoint of the targets for the inputs (mains electricity) womdpw
            # of data.
            #
            tar = targets[excerpt + self.offset].reshape(-1, 1)
            # The inp numpy array has batchsize rows.  Each row has window size number of readings.
            # The tar numpy array is the center point value of the target column within the window size
            # number of aggregate readings of the inp array.
            return inp, tar
#
# Each index will be the starting element within the inputs for a batch of data.
# getting the data will start with this element and read the next windowsize elements.
# Triggered once at the beginning as well as at the end of each epoch.
#

    def on_epoch_end(self):
        self.offset = int(0.5*(self.windowsize-1.0))
        self.max_batchsize = self.inputs.size - 2 * offset
        if self.batchsize < 0:
            self.batchsize = max_batchsize
        self.indexes = np.arange(max_batchsize)
        # From https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        # Shuffling the order in which examples are fed to the classifier is helpful so that
        # batches between epochs do not look alike. Doing so will eventually make our model more robust.
        if self.shuffle:
            np.random.shuffle(self.indexes)
