import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 上記のip2vecクラスとPreprocessクラスの定義は省略

# データの準備
flows = [["flow1_word1", "flow1_word2", "flow1_word3"], ["flow2_word1", "flow2_word2", "flow2_word3"]]
word_to_id = {"flow1_word1": 0, "flow1_word2": 1, "flow1_word3": 2, "flow2_word1": 3, "flow2_word2": 4, "flow2_word3": 5}
vocab_size = len(word_to_id)
emb_size = 100

# トレーニングデータの生成
preprocessor = Preprocess(flows, word_to_id, window_size=2)
X_train, y_train_raw = preprocessor.generate_training_data()

# y_trainをOne-Hotエンコーディング
encoder = OneHotEncoder(sparse=False, categories=[range(vocab_size)])
y_train = []
for context_words in y_train_raw:
    y_train.append(encoder.fit_transform(np.array(context_words).reshape(-1, 1)).sum(axis=0))
y_train = np.array(y_train).T

# モデルの初期化
model = ip2vec(flows, word_to_id, vocab_size, emb_size)

# 学習
learning_rate = 0.05
epochs = 1000
batch_size = 256
parameters = model.skipgram_model_training(X_train, y_train, vocab_size, emb_size, learning_rate, epochs, batch_size)

print("Training completed!")
