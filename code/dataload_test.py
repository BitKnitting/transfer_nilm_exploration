

import pandas as pd
import numpy as np
filename = 'created_data/microwave_validation_.csv'
data_frame = pd.read_csv(filename,
                         nrows=5,
                         names=['aggregate', 'microwave']
                         )

print(data_frame.head())
np_array = np.array(data_frame)
inputs, targets = np_array[:, 0], np_array[:, 1]
print(inputs, targets)
