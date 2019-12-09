
import pandas as pd
from urllib.request import urlretrieve
###########################################
# Take a zipped pickle file and return a
# Pandas dataframe.  More description can
# be found here: http://bit.ly/2rButfh
###########################################


def get_dataframe_from_pkl_zip(url):
    file = 'file.zip'
    zipped_file = urlretrieve(url, file)
    filepath = '/content/'+file
    return pd.read_pickle(filepath)
###########################################
# Print interesting stats
###########################################


def print_stats(df, title):
    print(title)
    print('**************************')
    print('Start index: {}'.format(df.index.min()))
    print('--------------------------')
    print('End index: {}'.format(df.index.max()))
    print('--------------------------')
    print('Rank: {}'.format(len(df.shape)))
    print('--------------------------')
    print('Shape: {}'.format(df.shape))
    print('--------------------------')
    print('Data types: \n{}'.format(df.dtypes))
    print('--------------------------')
    print('Number of missing values:\n{}'.format(df.isnull().sum()))
    print('--------------------------')
    print('Summary Stats:\n {}'.format(df.describe()))
    print('--------------------------')
    print('Head:\n {}'.format(df.head()))
    print('--------------------------')
