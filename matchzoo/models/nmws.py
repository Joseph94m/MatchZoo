# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,merge, Dot
from keras.optimizers import Adam
from model import BasicModel

from utils.utility import *

class NMWS(BasicModel):
    def __init__(self, config):
        super(NMWS, self).__init__(config)
        self.__name = 'NMWS'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'layer_size', 'layer_count', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[NMWS] parameter check wrong')
        print('[NMWS] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('layer_size', 32)
        self.set_default('layer_count', 2)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        psi = Concatenate(axis=1) ([q_embed, d_embed])
        for layer_count in self.config['layer_count']:
            psi = Dropout(rate=self.config['dropout_rate'])(psi)
            psi = Dense(self.config['layer_size'], activation='sigmoid')(psi)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(psi)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(psi)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        return model
