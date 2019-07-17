from utils import *
import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np


def rnn_encoder(rnn_type, x, mask, d_dim, n_layer, initializer, pooling=None):
    seq_len = tf.reduce_sum(mask, 1)
    with tf.variable_scope('rnn_{}'.format(0)):
        h_outs, h_final = rnn_encoder_single_layer(
                    rnn_type,
                    x,
                    seq_len,
                    d_dim,
                    initializer)
    if n_layer > 1:
        for i in range(1, n_layer):
            with tf.variable_scope('rnn_{}'.format(i)):
                h_outs, h_final = rnn_encoder_single_layer(
                            rnn_type,
                            h_outs,
                            seq_len,
                            d_dim,
                            initializer)

    if pooling is None:
        h = h_outs
    else:
        if pooling == 'max':
            h = dynamic_max_pooling(h_outs, mask)
        elif pooling == 'last':
            h = h_final

    h_size = d_dim * (int(rnn_type.find('bi') > -1) + 1)
    return h, h_size, h_final


def dynamic_max_pooling(x, mask):
    mask = tf.expand_dims(tf.cast(mask, tf.float32) - 1., 2)
    return tf.reduce_max(x + mask * 999999, axis=1)


def rnn_encoder_single_layer(rnn_type, input, seq_len, d_dim, initializer):
    bi_directional = rnn_type.find('bi') == 0
    with tf.variable_scope('rnn_encoder'):
        batch_size = tf.shape(seq_len)[0]

        cell = {}
        initial_state = {}
        for d in ['forward', 'backward'] if bi_directional else ['forward']:
            with tf.variable_scope(d):
                cell[d] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
                                            d_dim,
                                            forget_bias=1.0,
                                            initializer=initializer,
                                            state_is_tuple=True)
                
                i_cell = tf.get_variable(d + 'i_cell',
                                         shape=[1, d_dim],
                                         dtype=tf.float32,
                                         initializer=initializer)
                i_output = tf.get_variable(d + 'i_output',
                                           shape=[1, d_dim],
                                           dtype=tf.float32,
                                           initializer=initializer)
                c_states = tf.tile(i_cell, tf.stack([batch_size, 1]))
                h_states = tf.tile(i_output, tf.stack([batch_size, 1]))
                initial_state[d] = tf.contrib.rnn.LSTMStateTuple(c_states,
                                                                 h_states)

        if bi_directional:
            raw_outputs, (fw, bw) = \
                tf.nn.bidirectional_dynamic_rnn(cell['forward'],
                                                cell['backward'],
                                                input,
                                                dtype=tf.float32,
                                                sequence_length=seq_len,
                                                initial_state_fw=initial_state['forward'],
                                                initial_state_bw=initial_state['backward'])
            raw_outputs = tf.concat(raw_outputs, axis=2)
            final_states = tf.concat([fw.h, bw.h], axis=1)

        else:
            raw_outputs, final_states = \
                tf.nn.dynamic_rnn(cell['forward'],
                                  input,
                                  dtype=tf.float32,
                                  sequence_length=seq_len,
                                  initial_state=initial_state['forward'])
            final_states = final_states.h
    return raw_outputs, final_states


def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def ffnn(inputs, h_dim, out_dim, n_layer, initializer,
         act_mid=tf.nn.tanh, act_last=tf.nn.tanh, use_bias=True,
         scope_name='ffnn'):
    h = inputs
    for i in range(n_layer-1):
        with tf.variable_scope('{}_{}'.format(scope_name, i)):
            w = tf.get_variable('W',
                                shape=[h_dim, h_dim],
                                initializer=initializer)
            if len(inputs.get_shape().as_list()) == 3:
                w = tf.tile(tf.expand_dims(w, axis=0),
                            [tf.shape(inputs)[0], 1, 1])
            h = tf.matmul(h, w)
            if use_bias:
                b = tf.get_variable('b',
                                    shape=[h_dim],
                                    initializer=tf.zeros_initializer())
                h = h + b
            if act_mid is not None:
                h = act_mid(h)

    with tf.variable_scope('{}_{}'.format(scope_name, n_layer-1)):
        W = tf.get_variable('W',
                            shape=[h_dim, out_dim],
                            initializer=initializer)
        if len(inputs.get_shape().as_list()) == 3:
            W = tf.tile(tf.expand_dims(W, axis=0), [tf.shape(inputs)[0], 1, 1])
        y_raw = tf.matmul(h, W)
        if use_bias:
            b = tf.get_variable('b',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer())
            y_raw = y_raw + b
        if act_last is not None:
            y_raw = act_last(y_raw)
    return y_raw


def create_mixed_trainable_emb(dim, n_ws, n_special_ws, initializer,
                               is_trainable, scope_name):
    """ Reserve index 0 for non-trainable padding, following by
    n_ws pretrained embeddings and n_special_ws trainable embeddings.
    """
    with tf.variable_scope(scope_name):
        pad_e = tf.get_variable(
            "pad_e",
            dtype=tf.float32,
            shape=[1, dim],
            initializer=tf.zeros_initializer(),
            trainable=False)

        e = tf.get_variable(
            "e",
            dtype=tf.float32,
            shape=[n_ws, dim],
            initializer=initializer,
            trainable=is_trainable)

        special_e = tf.get_variable(
            "special_e",
            dtype=tf.float32,
            shape=[n_special_ws, dim],
            initializer=initializer,
            trainable=True)

        mixed_e = tf.concat([pad_e, e, special_e], axis=0)
    return mixed_e, e


def dropout(x, keep_prob, is_train):
    if is_train and keep_prob > 0:
        x = tf.nn.dropout(x, keep_prob)
    return x


class BaseModel:
    def create_holder(self):
        self.x = tf.placeholder(tf.int32, [None, None, None])
        self.x_mask = tf.placeholder(tf.int32, [None, None, None])
        self.xw = tf.placeholder(tf.int32, [None, None])
        self.xw_mask = tf.placeholder(tf.int32, [None, None])

        self.y = tf.placeholder(tf.int32, [None, None])
        self.drp_keep = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

    def __init__(self, params, session):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.params = params
        self.session = session
        self.ignored_vars = []

    def build_graph(self):
        pass

    def get_hybrid_emb(self, x, x_mask, xw, ce, we, initializer):
        with tf.variable_scope('hybrid_emb'):
            dims = shape_list(x)
            n_w, cpw = dims[-2], dims[-1]
            dims_1 = reduce(mul, dims[:-1], 1)
            dims_2 = reduce(mul, dims[:-2], 1)

            x_flat = tf.reshape(x, (dims_1, cpw))
            x_mask_flat = tf.reshape(x_mask, (dims_1, cpw))
            x_rep = tf.nn.embedding_lookup(ce, x_flat)

            with tf.variable_scope('c_encoder'):
                xc_rep, xc_size, _ = rnn_encoder(
                        self.params['c_rnn_type'],
                        x_rep,
                        x_mask_flat,
                        self.params['c_h_dim'],
                        self.params['c_rnn_layers'],
                        initializer,
                        self.params['c_pooling'])
                xc_rep = tf.reshape(xc_rep, [dims_2, n_w, xc_size])

            xw_fat = tf.reshape(xw, (dims_2, n_w))
            xw_rep = tf.nn.embedding_lookup(we, xw_fat)
            w_rep = tf.concat([xc_rep, xw_rep], axis=-1)
            hw_size = self.params['we_dim'] + xc_size
        return w_rep, hw_size

    def encode(self, x, x_mask, xw, xw_mask,
               drp_keep, initializer, reuse):
        with tf.variable_scope('encoder', reuse=reuse):
            n_ce = max(self.params['c2id'].values()) + 1
            ce, _ = create_mixed_trainable_emb(
                        self.params['ce_dim'],
                        n_ce - len(RESERVE_TKS),
                        len(RESERVE_TKS) - 1,
                        initializer,
                        True,
                        'ce'
                        )

            n_we = max(self.params['w2id'].values()) + 1
            assert n_we == len(self.params['w2id'])
            we, we_core = create_mixed_trainable_emb(
                        self.params['we_dim'],
                        n_we - len(RESERVE_TKS),
                        len(RESERVE_TKS) - 1,
                        initializer,
                        self.params['we_trainable'],
                        'we'
                        )

            self.we = we
            self.we_core = we_core
            self.ignored_vars.append(we_core)

            with tf.variable_scope('main_text'):
                w_rep, _ = self.get_hybrid_emb(x, x_mask, xw,
                                               ce, we, initializer)
                w_rep = tf.nn.dropout(w_rep, drp_keep)

            with tf.variable_scope('w_encoder'):
                if self.params['w_encoder'] == 'rnn':
                    hw, hw_size, h_last = rnn_encoder(
                                        self.params['w_rnn_type'],
                                        w_rep,
                                        xw_mask,
                                        self.params['w_h_dim'],
                                        self.params['w_rnn_layers'],
                                        initializer,
                                        None)
        return hw, hw_size, h_last

    def hw_pooling(self, hw, mask, hw_size, out_size, use_l2, initializer,
                   scope_name, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            h = dynamic_max_pooling(hw, mask)

            with tf.variable_scope('ff0', reuse=reuse):
                h = ffnn(
                        h, hw_size, out_size,
                        1, initializer)
            if use_l2:
                h_ = tf.nn.l2_normalize(h, -1)
            else:
                h_ = h
        return h_, h

    def fill_token_idx(self, words_batch, b_size, mlen, cpw):
        x = np.ones((b_size, mlen, cpw)) * self.params['c2id'][PAD]
        x_mask = np.zeros((b_size, mlen, cpw))
        xw = np.ones((b_size, mlen)) * self.params['w2id'][PAD]
        xw_mask = np.zeros((b_size, mlen))

        for i, words in enumerate(words_batch):
            c_ids = np.ones((mlen, cpw)) * self.params['c2id'][PAD]
            c_mask = np.zeros((mlen, cpw))
            for j, w in enumerate([[START_S]] + words[:mlen-2] + [[END_S]]):
                tmp = [self.params['c2id'][c]if c in self.params['c2id'] else self.params['c2id'][UNK]
                       for c in [START_W] + list(w)[:self.params['max_c_per_w']]+[END_W]]
                c_ids[j, :len(tmp)] = tmp
                c_mask[j, :len(tmp)] = 1
                if w[0] in RESERVE_TKS:
                    w_ = w[0]
                elif self.params['we_is_lw']:
                    w_ = w.lower()
                else:
                    w_ = w
                if w_ in self.params['w2id']:
                    xw[i, j] = self.params['w2id'][w_]
                else:
                    if self.params['try_lw_emb'] and w_.lower() in self.params['w2id']:
                        xw[i, j] = self.params['w2id'][w_.lower()]
                    else:
                        xw[i, j] = self.params['w2id'][UNK]
            x[i] = c_ids
            x_mask[i] = c_mask
            xw_mask[i, :len(words) + 2] = 1
        return x, x_mask, xw, xw_mask


class NameEncoder(BaseModel):
    def __init__(self, params, session):
        super().__init__(params, session)
        self.cf = params['tasks']['diff_name']
        self.params = params
        self.build_graph()

    def build_graph(self):
        self.create_holder()
        hw, hw_size, h_last = self.encode(self.x,
                                    self.x_mask,
                                    self.xw,
                                    self.xw_mask,
                                    self.drp_keep,
                                    self.initializer,
                                    reuse=tf.AUTO_REUSE)
        h, h_raw = self.hw_pooling(hw,
                                   self.xw_mask,
                                   hw_size,
                                   self.params['w_h_dim'],
                                   self.cf['use_l2_norm'],
                                   self.initializer,
                                   scope_name='diff_name_classifier',
                                   reuse=tf.AUTO_REUSE)
        self.h = h

    def get_fd_data(self, data_batch):
        max_len = max([len(row) for row in data_batch]) + 2
        b_size = len(data_batch)
        chars_per_word = self.params['max_c_per_w'] + 2
        x, x_mask, xw, xw_mask = self.fill_token_idx(data_batch, b_size,
                                                     max_len, chars_per_word)
        data_dict = {
            self.x: x,
            self.x_mask: x_mask,
            self.xw: xw,
            self.xw_mask: xw_mask,
            self.drp_keep: 1.,
            }
        return data_dict
