import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
from tqdm import tqdm
import numpy as np
import random
import model
import preprocess
# from model import Skipgram

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self,w2v,v2w,freq,emb_dim):
        self.v2w = v2w
        self.w2v = w2v
        self.unigram_table = self.noise(w2v,freq)
        self.vocab_size = len(w2v)
        self.model = model.Skipgram(self.vocab_size,emb_dim).to(device)
       # self.model = model.Skipgram(self.vocab_size,emb_dim)
        self.optim = optim.Adam(self.model.parameters())
        self.best_loss = float('inf')  # 最初のbest_lossを無限大に設定
        self.patience = 0

    def noise(self,w2v, freq):
        unigram_table = []
        total_word = sum([c for c in freq.values()])
        for w,v in w2v.items():
            unigram_table.extend([v]*int(((freq[w]/total_word)**0.75)/0.001))
        return unigram_table

    def negative_sampling(self,batch_size,neg_num,batch_target):
        neg = np.zeros((neg_num))
       # print("neg:", neg)
       # for i in range(batch_size):
        for i in range(len(batch_target)):
            sample = random.sample(self.unigram_table, neg_num)
           # print("sample:", sample)
            while batch_target[i] in sample:
                sample = random.sample(self.unigram_table, neg_num)
            neg = np.vstack([neg,sample])
        return neg[1:batch_size+1]

    def fit(self,data,max_epoch,batch_size,neg_num, patience_limit=4, save_every=1, save_path='model_checkpoint'):
        run_losses = []
        for epoch in range(max_epoch):
            run_loss = 0

           # print(data)
            for batch in tqdm(data):

                batch = np.array(batch)  # batchをlistからnumpyのndarrayに変換
               # print("batch:", batch)
                context,target = batch[:,1],batch[:,0]
               # print("context =", context,"target =" , target)
                self.optim.zero_grad()
                batch_neg = self.negative_sampling(batch_size,neg_num,target)
                context = V(th.LongTensor(context)).to(device)
                target = V(th.LongTensor(target)).to(device)
                batch_neg = V(th.LongTensor(batch_neg.astype(int))).to(device)
               # context = V(th.LongTensor(context))
               # target = V(th.LongTensor(target))
               # batch_neg = V(th.LongTensor(batch_neg.astype(int)))

                loss = self.model(target, context, batch_neg)
                loss.backward()
                self.optim.step()
                run_loss += loss.cpu().item()
            
            # 指定された間隔でモデルを保存
            if (epoch + 1) % save_every == 0:
                self.save_model(epoch, save_path)
            run_losses.append(run_loss/len(data))
            print("epoch:", epoch,"run_loss:", run_loss)
            if run_loss < self.best_loss:
                self.best_loss = run_loss
                self.patience = 0
            else:
                self.patience += 1
                if self.patience >= patience_limit:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break  # 早期終了

        return run_losses
    def save_model(self, epoch, save_path):
        model_state = self.model.state_dict()
        save_dict = {
            'model_state': model_state,
            'w2v': self.w2v,
            'v2w': self.v2w,
            'epoch': epoch
        }
        th.save(save_dict, f'{save_path}_epoch_{epoch}.pth')
        print(f'Model saved for epoch {epoch}')

    def most_similar(self,word,top):
        W = self.model.state_dict()["u_embedding.weight"]
        idx = w2v[word]
        similar_score = {}
        for i,vec in enumerate(W):
            if i != idx:
                d = vec.dot(W[idx])
                similar_score[self.v2w[i]] = d
        similar_score = sorted(similar_score.items(), key=lambda x: -x[1])[:top]
        for k,v in similar_score:
            print(k,":",round(v.item(),2))
