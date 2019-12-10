#
# The DataGenerator class is a rewrite of the DataProvider.py code
# found in this GitHub:
#
# My goals are:
# - understand how the data is fed into the model.
# - simplify with the assumption of running within a colab environment.
#
#
# df: the dataframe where the first column is the aggregate electricity readings and
# the second column are the device readings.  See create_trainset_redd.py, which creates
# the zipped pickled pandas files.
# batchsize: the number of rows of readings that will be returned during a next()
import pandas as pd
import numpy as np


class DataGenerator():
    def __init__(self, windowsize, batchsize,  shuffle):
        self.windowsize = windowsize
        self.shuffle = shuffle
        self.batchsize = batchsize

    def get_batch(self, df):
        np_array = np.array(df)
        inputs, targets = np_array[:, 0], np_array[:, 1]
        # Figure out how many batches can be made based on the length of the columns of data.
        offset = int(0.5*(self.windowsize-1.0))
        max_batchsize = inputs.size - 2 * offset
        if self.batchsize < 0:
            self.batchsize = max_batchsize
        indices = np.arange(max_batchsize)
        # From https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        # Shuffling the order in which examples are fed to the classifier is helpful so that
        # batches between epochs do not look alike. Doing so will eventually make our model more robust.
        if self.shuffle:
            np.random.shuffle(indices)
        # excerpt gets us batchsize rows.  Each row has windowsize number of readings.  The readings are in order
        # The researchers note "sliding window" here where I think of it more as a random place to start grabbing
        # a windowsize bunch of readings.
        for start_idx in range(0, max_batchsize, self.batchsize):
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
            tar = targets[excerpt + offset].reshape(-1, 1)
            # The inp numpy array has batchsize rows.  Each row has window size number of readings.
            # The tar numpy array is the center point value of the target column within the window size
            # number of aggregate readings of the inp array.
            yield inp, tar
