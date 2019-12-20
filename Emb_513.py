from keras_bert import extract_embeddings
import numpy as np
import codecs
import pickle as pkl
output_prefix = './datasets/'
model_path ='/home/jyzhou/.keras/datasets/multi_cased_L-12_H-768_A-12/'
data_path_prefix = './datasets/'
i = 0
#data_prefix = 'uUs_split_''
train_data_path = data_path_prefix+'train_wv.pkl'
train_data_file = open(train_data_path,'rb')
train_vec = pkl.load(train_data_file)
train_texts = [' '.join(t) for t in train_vec]

train_embeddings = extract_embeddings(model_path,train_texts)
train_output_path = output_prefix+'train_bert'+'.npy'
train_sentence_emb = np.array([e[0] for e in train_embeddings])
np.save(train_output_path,train_sentence_emb)
'''
test_data_path = data_path_prefix+'test_wv.pkl'
test_data_file = open(test_data_path,'rb')
test_vec = pkl.load(test_data_file)
test_texts = [' '.join(t) for t in test_vec]
test_embeddings = extract_embeddings(model_path,test_texts)
test_output_path = output_prefix+'test_bert'+'.npy'
test_sentence_emb = np.array([e[0] for e in test_embeddings])
np.save(test_output_path,test_sentence_emb)





model = load_trained_model_from_checkpoint(config_path, checkpoint_path) 
pool_layer = MaskedGlobalMaxPool1D(name='Pooling')(model.output) 
model = keras.models.Model(inputs=model.inputs, outputs=pool_layer) 
model.summary(line_length=120) 
def get_sentence_emb(tokens):
    
    #tokens = ['[CLS]', '语', '言', '模', '型', '[SEP]'] 

    token_dict = {} 
    with codecs.open(dict_path, 'r', 'utf8') as reader: 
        for line in reader: 
            token = line.strip() 
            token_dict[token] = len(token_dict) 

  
    token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))]) 
    seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))]) 
    #print('Inputs:', token_input[0][:len(tokens)]) 
    predicts = model.predict([token_input, seg_input])[0] 
    
    #print('Pooled:', predicts.tolist()[:5]) 
    return predicts.tolist()[]

datafile = open(data_path,'r',encoding='utf-8')
data = datafile.readlines()
tmp = open('./data/concated/incase.txt','w')
i = 0
embeddings = []
while i < (len(data)-32):
    texts = data[i:i+32]
    print('processing ' + str(i) + '/'+str(len(data))+'......')
    this_emb = extract_embeddings(model_path,texts)[0]
    #embeddings.append(this_emb)
    tmp.write(str(this_emb))
    i += 32
'''
#np.save(output_path,embeddiings)


	

