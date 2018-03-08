
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
import re
from keras.models import Model, load_model
from keras.layers import Input


# ## Read data

# In[5]:

df_seq = pd.read_csv('../data/cat113_root_name.csv')
X = df_seq.iloc[:,0].values
y = df_seq.iloc[:,1].values
X_roots = df_seq.iloc[:,2].values


# In[6]:

# pre-processing
def pre_process(X):
    X_p = []

    for name in X:
        name = name.lower().split()
        name = [re.compile('[(),]+').sub('', w) for w in name] 
        name = [w for w in name if re.compile('[\W_]+').sub('', w)] # remove all words that only constain special character
        name = ' '.join(name)
        #name = ViTokenizer.tokenize(name)
        X_p.append(name)

    return X_p


# In[7]:

X = pre_process(X)
X_roots = pre_process(X_roots)


# In[8]:

num_samples = len(X)
input_texts = []
target_texts = []

input_tokens = set()
target_tokens = set()

for i in range(num_samples):
    # cast into tokens
    input_texts.append(X[i])
    target_texts.append('\t' + X_roots[i] + '\n')
    
    for word in input_texts[i]:
        if word not in input_tokens:
            input_tokens.add(word)
    for word in target_texts[i]:
        if word not in target_tokens:
            target_tokens.add(word)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

max_encoder_seq_length = max([len(seq) for seq in input_texts])
max_decoder_seq_length = max([len(seq) for seq in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)  


# In[9]:

# building dictionary of tokens
input_token_index = dict([(token, i) for i, token in enumerate(input_tokens)])
target_token_index = dict([(token, i) for i, token in enumerate(target_tokens)])


# ### Read test data

# In[ ]:

df_seq = pd.read_csv('../data/cat113_root_name_test.csv')
X = df_seq.iloc[:,0].values
X_roots = df_seq.iloc[:,2].values

X = pre_process(X)
X_roots = pre_process(X_roots)

input_texts = []
target_texts = []

for i in range(len(X)):
    # cast into tokens
    input_texts.append(X[i])
    target_texts.append(X_roots[i])


# In[14]:

# building embedding for input and target data
encoder_input_data = np.zeros((num_samples, max_encoder_seq_length, num_encoder_tokens))


# In[15]:

for i, input_text in enumerate(input_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0


# ## Load trained model

# In[2]:

latent_dim = 256  # Latent dimensionality of the encoding space.
# Restore the model and construct the encoder and decoder.
model = load_model('s2s.h5')

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# ## Test predict

# In[ ]:

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


# In[ ]:

# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# In[ ]:

for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Root sentence:', target_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

