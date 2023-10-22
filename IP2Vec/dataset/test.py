import sys
import os
sys.path.append('..')

import pandas as pd

dataset_dir = os.path.dirname(os.path.abspath(__file__))

file_name = 'modified_botnet_first10rows.csv'
file_path = dataset_dir + '/' + file_name

df = pd.read_csv(file_path)

for flow in df.itertuples(index=True, name='Pandas'):
    for i in range(len(flow)-1): 
        print(flow[i+1])




        """
        center_word = self.word_to_id[flow[i]] 
        for j in range(i-self.window_size,i+self.window_size+1):
                if (i==1 and j==0)|(i==2 and j==0):
                    continue
                if i!=j and j>=0 and j<len(flow):
                    context = word_to_id[flow[j]]
                    X_train.append(center_word)
                    y_train.append(context)
        X_train = np.expand_dims(X_train, axis=0) 
        y_train = np.expand_dims(y_train, axis=0) 
   
        return X_train, y_train
        """