from python_lib.Seq2PointGenerator import Seq2PointGenerator
import pandas as pd
import numpy as np

a = np.arange(20)
a = a.reshape((10, 2))
df = pd.DataFrame(a)
df.columns = ['aggregate', 'microwave']
gen = Seq2PointGenerator(df,windowsize=3, batchsize=5)
