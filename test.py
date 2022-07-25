import torch
import gensim

pretrained_model = gensim.models.KeyedVectors.load_word2vec_format('./dataset/pretrain/law_token_vec_300.bin', binary=False)
print(pretrained_model.key_to_index[''])

