###################################################################################
# Title  : KSE526 project baseline
# Author : hs_min
# Date   : 2020.11.25
###################################################################################
#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import RNN, GRU, BatchNormalization, Dropout, TimeDistributed, Softmax, Dot, Bidirectional, Layer, Conv1D, MaxPooling1D, Flatten, RepeatVector, LSTM, Attention, Concatenate, Dense
import tensorflow.keras.backend as K
#%%
###################################################################
# Loss
###################################################################
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))

def weighted_root_mean_squared_error(y_true, y_pred):#, w):
    w = 0.2
    mask = tf.cast(tf.less(y_pred, y_true), dtype=tf.float64)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred))) + mask * w * (y_true-y_pred)

def last_time_step_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true[:,-1], y_pred[:,-1])


###################################################################
# Model
###################################################################
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query) : # 단, key와 value는 같음
    # query shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
#%%
class MyLayer(Layer):
    def __init__(self, config):
        super(MyLayer, self).__init__()
class MyModel(Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
class CNNBiLSTMATTN_noAuxs(Model):
    def __init__(self, config):
        super(CNNBiLSTMATTN_noAuxs, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units

        self.conv1d1 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.conv1d2 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        
        self.mp1d = MaxPooling1D(pool_size = 2)        
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))
        # self.rv = RepeatVector(self.n_outputs)
        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)#, activation='relu')

        self.conv1d3 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.aux_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.5, 
                                        return_sequences=True, return_state = True))

        self.aux_attention = BahdanauAttention(self.lstm_units)
        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1d1(inputs[0])
        x = self.conv1d2(x)
        x = self.mp1d(x)
        # encoder_lstm = self.lstm1(x)
        # encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm1(x)
        # encoder_lstm, forward_h,  backward_h  = self.lstm1(inputs[0])
        # state_h = self.concat([forward_h, backward_h])
        # decoder_input = self.rv(state_h)
        decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm2(x)
        # decoder_lstm, forward_h, backward_h = self.lstm2(x)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        # state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(decoder_lstm, state_h)
        x = self.fcn1(context_vector)
        # x = self.dropout(x)
        
        x_aux1 = self.conv1d3(inputs[1])
        aux_lstm, aux_forward_h, aux_forward_c, aux_backward_h, aux_backward_c = self.aux_lstm(x_aux1)
        aux_state_h = self.concat([aux_forward_h, aux_backward_h]) # 은닉 상태
        aux_context_vector, aux_attention_weights = self.aux_attention(aux_lstm, aux_state_h)
        x_aux1 = self.aux_fcn1(aux_context_vector)

        x_aux2 = self.aux_fnc2(inputs[2])
        x_aux2 = self.aux_flatten(x_aux2)

        # x = self.concat([x]#, x_aux1, x_aux2]) 
        x = self.fcn3(x)
        x = self.fcn4(x)
        
        return x

class CNNBiLSTMATTN_noAUX1(Model):
    def __init__(self, config):
        super(CNNBiLSTMATTN_noAUX1, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units

        self.conv1d1 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.conv1d2 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        
        self.mp1d = MaxPooling1D(pool_size = 2)        
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))
        # self.rv = RepeatVector(self.n_outputs)
        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)#, activation='relu')

        self.conv1d3 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.aux_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.5, 
                                        return_sequences=True, return_state = True))

        self.aux_attention = BahdanauAttention(self.lstm_units)
        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1d1(inputs[0])
        x = self.conv1d2(x)
        x = self.mp1d(x)
        # encoder_lstm = self.lstm1(x)
        # encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm1(x)
        # encoder_lstm, forward_h,  backward_h  = self.lstm1(inputs[0])
        # state_h = self.concat([forward_h, backward_h])
        # decoder_input = self.rv(state_h)
        decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm2(x)
        # decoder_lstm, forward_h, backward_h = self.lstm2(x)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        # state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(decoder_lstm, state_h)
        x = self.fcn1(context_vector)
        # x = self.dropout(x)
        
        x_aux1 = self.conv1d3(inputs[1])
        aux_lstm, aux_forward_h, aux_forward_c, aux_backward_h, aux_backward_c = self.aux_lstm(x_aux1)
        aux_state_h = self.concat([aux_forward_h, aux_backward_h]) # 은닉 상태
        aux_context_vector, aux_attention_weights = self.aux_attention(aux_lstm, aux_state_h)
        x_aux1 = self.aux_fcn1(aux_context_vector)

        x_aux2 = self.aux_fnc2(inputs[2])
        x_aux2 = self.aux_flatten(x_aux2)

        x = self.concat([x,  x_aux2]) # x_aux1,
        x = self.fcn3(x)
        x = self.fcn4(x)
        
        return x

class CNNBiLSTMATTN_noAUX2(Model):
    def __init__(self, config):
        super(CNNBiLSTMATTN_noAUX2, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units

        self.conv1d1 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.conv1d2 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        
        self.mp1d = MaxPooling1D(pool_size = 2)        
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))
        # self.rv = RepeatVector(self.n_outputs)
        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)#, activation='relu')

        self.conv1d3 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.aux_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.5, 
                                        return_sequences=True, return_state = True))

        self.aux_attention = BahdanauAttention(self.lstm_units)
        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1d1(inputs[0])
        x = self.conv1d2(x)
        x = self.mp1d(x)
        # encoder_lstm = self.lstm1(x)
        # encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm1(x)
        # encoder_lstm, forward_h,  backward_h  = self.lstm1(inputs[0])
        # state_h = self.concat([forward_h, backward_h])
        # decoder_input = self.rv(state_h)
        decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm2(x)
        # decoder_lstm, forward_h, backward_h = self.lstm2(x)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        # state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(decoder_lstm, state_h)
        x = self.fcn1(context_vector)
        # x = self.dropout(x)
        
        x_aux1 = self.conv1d3(inputs[1])
        aux_lstm, aux_forward_h, aux_forward_c, aux_backward_h, aux_backward_c = self.aux_lstm(x_aux1)
        aux_state_h = self.concat([aux_forward_h, aux_backward_h]) # 은닉 상태
        aux_context_vector, aux_attention_weights = self.aux_attention(aux_lstm, aux_state_h)
        x_aux1 = self.aux_fcn1(aux_context_vector)

        x_aux2 = self.aux_fnc2(inputs[2])
        x_aux2 = self.aux_flatten(x_aux2)

        x = self.concat([x, x_aux1])#, x_aux2])
        x = self.fcn3(x)
        x = self.fcn4(x)
        
        return x

# %%
class CNNBiLSTMATTN(Model):
    def __init__(self, config):
        super(CNNBiLSTMATTN, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units

        self.conv1d1 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.conv1d2 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)
        
        self.mp1d = MaxPooling1D(pool_size = 2)        
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))
        self.flatten = Flatten()
        self.rv = RepeatVector(self.n_outputs)
        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)#, activation='relu')

        self.conv1d3 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        # self.aux_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.5, 
        #                                 return_sequences=True, return_state = True))

        # self.aux_attention = BahdanauAttention(self.lstm_units)
        # self.aux_fcn1 = Dense(20)
        self.aux_fcn1 = TimeDistributed(Dense(20))
        self.aux_flatten1 = Flatten()
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten2 = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

        self.is_x_aux1 = config.is_x_aux1
        self.is_x_aux2 = config.is_x_aux2

    def call(self, inputs):
        x = self.conv1d1(inputs[0])
        x = self.conv1d2(x)
        x = self.mp1d(x)
        # encoder_lstm = self.lstm1(x)
        # encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm1(x)
        # encoder_lstm, forward_h,  backward_h  = self.lstm1(inputs[0])
        # state_h = self.concat([forward_h, backward_h])
        decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm2(x)
        # decoder_lstm, forward_h, backward_h = self.lstm2(x)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        # state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(x, state_h)
        x = self.fcn1(context_vector)
        # x = self.dropout(x)
        
        x_aux1 = self.conv1d3(inputs[1])
        aux_lstm, aux_forward_h, aux_forward_c, aux_backward_h, aux_backward_c = self.aux_lstm(x_aux1)
        # aux_state_h = self.concat([aux_forward_h, aux_backward_h]) # 은닉 상태
        # aux_context_vector, aux_attention_weights = self.aux_attention(aux_lstm, aux_state_h)
        # x_aux1 = self.aux_fcn1(aux_context_vector)
        x_aux1 = self.aux_fcn1(x_aux1)
        x_aux1 = self.aux_flatten1(x_aux1)

        x_aux2 = self.aux_fnc2(inputs[2])
        x_aux2 = self.aux_flatten2(x_aux2)
        
        if self.is_x_aux1 :
            print("aux1",inputs[1].shape)
            x = self.concat([x, x_aux1])
        if self.is_x_aux2 :
            print("aux2",inputs[2].shape)
            x = self.concat([x, x_aux2])
        # x = self.concat([x, x_aux2])

        x = self.fcn3(x)
        x = self.fcn4(x)
        return x#, attention_weights
#%%
class BiLSTMATTN(Model):
    def __init__(self, config):
        super(BiLSTMATTN, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))
        # self.rv = RepeatVector(self.n_outputs)
        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)#, activation='relu')

        self.aux_lstm = LSTM(self.lstm_units, dropout=0.2, return_sequences=False)
        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        encoder_lstm = self.lstm1(inputs[0])
        # encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm(inputs[0])
        # encoder_lstm, forward_h,  backward_h  = self.encoder_lstm(inputs[0])
        # state_h = self.concat([forward_h, backward_h])
        # decoder_input = self.rv(state_h)
        decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm2(encoder_lstm)
        # decoder_lstm, forward_h, backward_h = self.decoder_lstm(encoder_lstm)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        # state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(encoder_lstm, state_h)
        x = self.fcn1(context_vector)
        # x = self.dropout(x)
        
        x_aux1 = self.aux_lstm(inputs[1])
        x_aux1 = self.aux_fcn1(x_aux1)

        x_aux2 = self.aux_fnc2(inputs[2])
        x_aux2 = self.aux_flatten(x_aux2)

        x = self.concat([x, x_aux1, x_aux2])
        x = self.fcn3(x)
        x = self.fcn4(x)
        
        return x
#%%
#%%
class BiLSTMATTNre(Model):
    def __init__(self, config):
        super(BiLSTMATTNre, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units
        
        self.encoder_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.1, return_sequences = True, return_state= True, recurrent_initializer='glorot_uniform'))
        self.rv = RepeatVector(self.n_outputs)
        self.decoder_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.1, return_sequences = False, return_state=False, recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate(axis=-1)
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn0 = TimeDistributed(Dense(1))
        self.flatten = Flatten()

        self.fcn1 = Dense(50)#, activation='relu')

        self.aux_lstm = LSTM(self.lstm_units, dropout=0.5, return_sequences=False)
        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm(inputs[0])
        # encoder_lstm, forward_h,  backward_h  = self.encoder_lstm(inputs[0])
        state_h = self.concat([forward_h, backward_h])

        context_vector, attention_weights = self.attention(state_h, encoder_lstm) # state_h : query, state_h : values
        context_vector = self.rv(context_vector)
        decoder_input = self.concat([context_vector, inputs[2]])
        decoder_lstm = self.decoder_lstm(decoder_input)
        decoder_lstm = self.fcn0(decoder_lstm)
        decoder_lstm = self.flatten(decoder_lstm)

        x_aux1 = self.aux_lstm(inputs[1])
        x_aux1 = self.aux_fcn1(x_aux1)

        # x_aux2 = self.aux_fnc2(inputs[2])
        # x_aux2 = self.aux_flatten(x_aux2)

        x = self.concat([decoder_lstm, x_aux1])#, x_aux2])
        x = self.fcn3(x)
        x = self.fcn4(x)
        
        return x 
#%%
class LSTMaux(Model):
    def __init__(self, config):
        super(LSTMaux, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units

        self.lstm_encoder = LSTM(units = self.lstm_units, dropout=0.5, return_sequences = True, return_state= True)
        # output, forward_h, backward_h, forward_c, backward_c
        self.lstm_decoder = LSTM(units = self.lstm_units, dropout=0.5, return_sequences = True, return_state = False)
        # self.td1 = TimeDistributed(Dense(10, activation = self.activation ))
        self.flatten = Flatten()

        self.aux_dense = TimeDistributed(Dense(1))
        self.aux_concat = Concatenate()
        
        self.outputs = Dense(self.n_outputs) # self.n_outputs

    def call(self, inputs):
        encoder_stack_h, encoder_last_h, encoder_last_c = self.lstm_encoder(inputs[0])
        decoder_input = self.rv(encoder_last_h)
        decoder_stack_h = self.lstm_decoder(decoder_input, initial_state = [encoder_last_h, encoder_last_c])
        decoder_flatten = self.flatten(decoder_stack_h)

        aux_input = self.aux_dense(inputs[1])
        aux_flatten = self.flatten(aux_input)
        aux_concat = self.aux_concat([decoder_flatten, aux_flatten])

        outputs = self.outputs(aux_concat)
        return outputs
#%%
class LSTMATTN(Model):
    def __init__(self, config):
        super(LSTMATTN, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units

        self.lstm_encoder = LSTM(units = self.lstm_units, return_sequences = True, return_state= True)
        self.rv = RepeatVector(self.n_outputs)
        # output, forward_h, backward_h, forward_c, backward_c
        self.lstm_decoder = LSTM(units = self.lstm_units, return_sequences = True, return_state = False)
        # self.td1 = TimeDistributed(Dense(10, activation = self.activation ))
        self.attention = Dot(axes=[2,2])
        self.softmax = Softmax()
        self.context = Dot(axes=[2,1])
        self.concat = Concatenate()
        self.flatten = Flatten()
        self.fcn = Dense(30) # self.n_outputs
        self.outputs = Dense(self.n_outputs) # self.n_outputs

    def call(self, inputs):
        # x = self.lstm_in(inputs)
        encoder_stack_h, encoder_last_h, encoder_last_c = self.lstm_encoder(inputs[0])
        decoder_input = self.rv(encoder_last_h)
        decoder_stack_h = self.lstm_decoder(decoder_input, initial_state = [encoder_last_h, encoder_last_c])
        attention = self.attention([decoder_stack_h, encoder_stack_h])

        attention = self.softmax(attention)
        context = self.context([attention, encoder_stack_h])
        decoder_combined_context = self.concat([context, decoder_stack_h])
        
        flatten = self.flatten(decoder_combined_context)
        
        fcn = self.fcn(flatten)
        
        outputs = self.outputs(fcn)
        return outputs
#%%
class CNNLSTMATTN(Model):
    def __init__(self, config
                        ):
        super(CNNLSTMATTN, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units

        self.conv1d1 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)
        self.conv1d2 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)
        self.mp1d = MaxPooling1D(pool_size = 2)
        self.flatten = Flatten()
        # self.lstm_in = LSTM(units = self.units, activation = self.activation)
        self.rv = RepeatVector(self.n_outputs)
        # output, forward_h, backward_h, forward_c, backward_c
        self.lstm_out = Bidirectional(LSTM(units = self.lstm_units, return_sequences = True, return_state = True))
        # self.td1 = TimeDistributed(Dense(10, activation = self.activation ))
        self.attention = Attention()
        self.concat = Concatenate()
        self.td2 = Dense(self.n_outputs) # self.n_outputs

    def call(self, inputs):
        # x = self.lstm_in(inputs)
        x = self.conv1d1(inputs)
        x = self.conv1d2(x)
        x = self.mp1d(x)
        x = self.flatten(x)
        x = self.rv(x)
        x = self.lstm_out(x)
        # x = self.td1(x)
        x = self.attention([x,x])
        x = self.td2(x)
        return tf.reshape(x, shape =(-1,self.n_outputs))

#%%
class CNNLSTM(Model):
    def __init__(self, config):
        super(CNNLSTM, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units

        self.conv1d1 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)
        self.mp1d = MaxPooling1D(pool_size = 2)
        self.flatten = Flatten()
        self.rv = RepeatVector(self.n_outputs)
        self.lstm = LSTM(units = self.lstm_units, return_sequences = False)
        self.fcn = Dense(self.n_outputs) # self.n_outputs

    def call(self, inputs):
        # x = self.lstm_in(inputs)
        x = self.conv1d1(inputs)
        x = self.mp1d(x)
        x = self.flatten(x)
        x = self.rv(x)
        x = self.lstm(x)
        x = self.fcn(x)
        return tf.keras.backend.reshape(x, shape = (-1,self.n_outputs))

#%%
class CNNs(Model):
    def __init__(self, config):
        super(CNNs, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units

        self.conv1d1 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.flatten = Flatten()
        self.d = Dense(self.n_outputs) # self.n_outputs

    def call(self, inputs):
        # x = self.lstm_in(inputs)
        x = self.conv1d1(inputs)
        x = self.flatten(x)
        x = self.d(x)
        return tf.keras.backend.reshape(x, shape =(-1,self.n_outputs))
#%%
class LSTMs(Model):
    def __init__(self, config):
        super(LSTMs, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.lstm_units = config.lstm_units

        self.lstm_encoder = LSTM(units = self.lstm_units, return_sequences = False, return_state= False)
        # self.flatten = Flatten()
        self.outputs = Dense(self.n_outputs) # self.n_outputs

    def call(self, inputs):
        encoder_stack_h = self.lstm_encoder(inputs)
        # flatten = self.flatten(encoder_stack_h)
        outputs = self.outputs(encoder_stack_h)
        return outputs
# %%
# class BiLSTMATTN(Model):
#     def __init__(self, config):
#         super(BiLSTMATTN, self).__init__()
#         self.n_outputs = config.label_width
#         self.filters = config.filters
#         self.kernel_size = config.kernel_size
#         self.activation = config.activation
#         self.lstm_units = config.lstm_units
#         self.attn_units = config.attn_units
        
#         self.encoder_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.1, return_sequences = True, return_state= True, recurrent_initializer='glorot_uniform'))
#         # self.rv = RepeatVector(self.n_outputs)
#         self.decoder_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.1, return_sequences = True, return_state=True, recurrent_initializer='glorot_uniform'))
#         self.concat = Concatenate()
#         self.attention = BahdanauAttention(self.attn_units)
#         self.fcn1 = Dense(50)#, activation='relu')

#         self.aux_lstm = LSTM(self.lstm_units, dropout=0.5, return_sequences=False)
#         self.aux_fcn1 = Dense(20)
        
#         self.aux_fnc2 = TimeDistributed(Dense(20))
#         self.aux_flatten = Flatten()

#         self.fcn3 = Dense(10)
#         self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

#     def call(self, inputs):
#         encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm(inputs[0])
#         # encoder_lstm, forward_h,  backward_h  = self.encoder_lstm(inputs[0])
#         # state_h = self.concat([forward_h, backward_h])
#         # decoder_input = self.rv(state_h)
#         decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.decoder_lstm(encoder_lstm)
#         # decoder_lstm, forward_h, backward_h = self.decoder_lstm(encoder_lstm)
#         state_h = self.concat([forward_h, backward_h]) # 은닉 상태
#         # state_c = self.concat([forward_c, backward_c])
#         context_vector, attention_weights = self.attention(encoder_lstm, state_h)
        
#         x = self.fcn1(context_vector)
#         # x = self.dropout(x)
        
#         x_aux1 = self.aux_lstm(inputs[1])
#         x_aux1 = self.aux_fcn1(x_aux1)

#         x_aux2 = self.aux_fnc2(inputs[2])
#         x_aux2 = self.aux_flatten(x_aux2)

#         x = self.concat([x, x_aux1, x_aux2])
#         x = self.fcn3(x)
#         x = self.fcn4(x)
        
#         return x