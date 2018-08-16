from __future__ import print_function
import socket
import signal
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)

from collections import OrderedDict

import keras
import keras.backend as K
from keras.models import Sequential, Model


from utils import *
import inputs
import metrics
from losses import *
from optimizers import *

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)


TCP_IP = "127.0.0.1"
TCP_PORT = 6776
ENCODING='utf-8'
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo




def predict(config):
    print(json.dumps(config, indent=2), end='\n')
    input_conf = config['inputs']
    share_input_conf = input_conf['share']
        # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys()), end='\n')
        # collect dataset identification
    
    
    output_conf = config['outputs']
    
    ######## Load Model ########
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])

    model = load_model(config)
    model.load_weights(weights_file)
    sock = socket.socket(socket.AF_INET, # Internet
                          socket.SOCK_STREAM) # UDP
    sock.bind((TCP_IP, TCP_PORT))

    
   
    
    print("Program is now ready for predictions")    
    
    while True:
        sock.listen(1)

        conn, addr = sock.accept()
        data = conn.recv(50000)
        data_string=str(data.decode(ENCODING))
        list_data=data_string.split("\n")
        list_list_data=[]
        for d in list_data:
            d_stripped=d.split(" ")
            if(len(d_stripped)>2):
                list_list_data.append((d_stripped[0],d_stripped[1],d_stripped[2]))
         
        conn.send("\n".encode(ENCODING))
        qData = conn.recv(1000) 
        qData_string=str(qData.decode(ENCODING))
        list_qData=qData_string.split("\n")
        list_list_qData={}
        dataset={}
        for d in list_qData:
            line=d.strip().split()
            tid=line[0]
            list_list_qData[tid]=list(map(int,line[2:])) 
        dataset['querydata']=list_list_qData
        conn.send("\n".encode(ENCODING))
        sizeData = conn.recv(50)
        sizeData_int = int(sizeData.decode(ENCODING))
        list_dData=[]
        for i in range(0,sizeData_int):
            conn.send("\n".encode(ENCODING))
            dData = conn.recv(50000) 
            dData_string=str(dData.decode(ENCODING))
            #print(dData_string)
            list_dData.append(dData_string)
        list_list_dData={}
        for d in list_dData:
            line=d.strip().split()
            tid=line[0]
            list_list_dData[tid]=list(map(int,line[2:])) 
        
        
        dataset['documentdata']= list_list_dData
        predict_gen = OrderedDict()
        for tag, conf in input_predict_conf.items():
            conf['data2'] = dataset['documentdata']
            conf['data1'] = dataset['querydata']
            generator = inputs.get(conf['input_type'])
            predict_gen[tag] = generator(
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf ,rel_data=list_list_data)
        dataset={}

        for tag, generator in predict_gen.items():
            genfun = generator.get_batch_generator()
            for input_data, y_true in genfun:
                y_pred = model.predict(input_data, batch_size=len(y_true) )
                print("Sending message")
               
                message = " ".join(map(str,y_pred.tolist()))
                message = message +'\n'
                #print(sys.getsizeof(message))
                #print(message)
                #sendSock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4112)  # Buffer size 8192
                conn.send(message.encode(ENCODING))
        
        

        

def main(argv):
    print("Server started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default='./models/arci.config', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    signal.signal(signal.SIGINT, signal_handler)
    with open(model_file, 'r') as f:
        config = json.load(f)
        predict(config)
    return




if __name__=='__main__':
    main(sys.argv)










