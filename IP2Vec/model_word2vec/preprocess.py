import numpy as np

class Preprocess:
    def __init__(self, flows, word_to_id, window_size):
        self.flows = flows
        self.word_to_id = word_to_id
        self.window_size = window_size

    def generate_training_data_v1(self):
        X_train = []
        y_train = []
     
        for flow in self.flows: 
            for i in range(len(flow)): 
                center_word = self.word_to_id[flow[i]]
                context_words = []
                for j in range(i-self.window_size, i+self.window_size+1):
                    if (i==1 and j==0) or (i==2 and j==0):
                        continue
                    if i != j and j >= 0 and j < len(flow):
                        context = self.word_to_id[flow[j]]
                        context_words.append(context)
                X_train.append(center_word)
                y_train.append(context_words)
        
        X_train = np.expand_dims(X_train, axis=0)
        return X_train, y_train

    def generate_training_data_v2(self):
        X_train = []
        y_train = []
        print(self.flows)
        for flow in self.flows: 
            print(flow)
            for i in range(len(flow)): 
                center_word = self.word_to_id[flow[i]] 
                for j in range(i-self.window_size,i+self.window_size+1):
                    if (i==1 and j==0)|(i==2 and j==0):
                        continue
                    if i!=j and j>=0 and j<len(flow):
                        # print(self.word_to_id)
                        # print(flow[j])
                        context = self.word_to_id[flow[j]]
                        X_train.append(center_word)
                        y_train.append(context)
        X_train = np.expand_dims(X_train, axis=0) 
        y_train = np.expand_dims(y_train, axis=0) 
   
        return X_train, y_train