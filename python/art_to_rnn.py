
# -----------------------------------------------------------------------------
# Adaptive Resonance Theory -> RNN Sequence Predictions
# Anthony R. Tatum
# 
# To Run:  You need to create a path for models to be save in.  Make it a new folder for this purpose only 
# e.g. path = "\\Desktop\\PythonScript\\learn\\mem\\"
# -----------------------------------------------------------------------------
# Reference: Adaptive Resonance Theory, Copyright (C) 2011 Nicolas P. Rougier
#            GIT : https://github.com/rougier/neural-networks/blob/master/art1.py
#
#            Grossberg, S. (1987)
#            Competitive learning: From interactive activation to
#            adaptive resonance, Cognitive Science, 11, 23-63
#
# -----------------------------------------------------------------------------

# This first section is an adaptation of Nicolas P. Rougier's ART model from github. 

from __future__ import print_function
from __future__ import division
import numpy as np
from numpy import array
from scipy import signal
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

# build path
root = os.getcwd()
path = "\\Desktop\\PythonScript\\learn\\mem\\"
directory = os.getcwd() + path

class ART:
    def __init__(self, n=5, m=10, rho=.5):
        '''
        Create network with specified shape
        Parameters:
        -----------
        n : int
            Size of input
        m : int
            Maximum number of internal units 
        rho : float
            Vigilance parameter
        '''
        # Comparison layer
        self.F1 = np.ones(n)
        # Recognition layer
        self.F2 = np.ones(m)
        # Feed-forward weights
        self.Wf = np.random.random((m,n))
        # Feed-back weights
        self.Wb = np.random.random((n,m))
        # Vigilance
        self.rho = rho
        # Number of active units in F2
        self.active = 0


    def learn(self, X):
        ''' Learn X '''

        # Compute F2 output and sort them (I)
        self.F2[...] = np.dot(self.Wf, X)
        I = np.argsort(self.F2[:self.active].ravel())[::-1]

        for i in I:
            # Check if nearest memory is above the vigilance level
            d = (self.Wb[:,i]*X).sum()/X.sum()
            if d >= self.rho:
                # Learn data
                self.Wb[:,i] *= X
                self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
                return self.Wb[:,i], i

        # No match found, increase the number of active units
        # and make the newly active unit to learn data
        if self.active < self.F2.size:
            i = self.active
            self.Wb[:,i] *= X
            self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
            self.active += 1
            return self.Wb[:,i], i

        return None,None


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    np.random.seed(1)

    # define train functions
    def sin_function(x):
        return np.sin(x)
    
    def rand_function(x):
        return np.random.randint(low=11, high=15, size=len(x))
    
    train_data_mult = 5
    
    # create data
    xaxis = np.arange(-train_data_mult, train_data_mult, .1)
    train_sin = sin_function(xaxis)
    train_rand = rand_function(xaxis)
    
    #plt.plot(train_sin)
    #plt.show()
    
    #plt.plot(train_rand)
    #plt.show()
    
    # join data together so network can loop through each element and develop clusters 
    train = [train_sin, train_rand, train_sin, train_rand, train_sin])
    
    samples = train 
    network = ART(100,10,rho=0.9)
    print(samples[0].sum())
    
    klus = []
    for i in range(len(samples)):
        Z, k = network.learn(samples[i])
        print("%c"%(ord('A')+i),"-> class",k)
        klus.append(k)
        klus_a = array(klus)
        k_cluster = klus_a.reshape((-1)) 
        #plt.plot(Z)
        #plt.show()

#############################################################################################################################################################
    
# This second section creates two Encoder-Decoder models, the best model is picked for each cluster from ART, saved and reused if that cluster is encountered again
    
    import keras
    from keras import backend as K
    from keras.models import load_model
    from keras.layers import Input, Dense, GRUCell, RNN, concatenate, TimeDistributed, Layer
    import tensorflow as tf
    from tensorflow.keras import initializers
    tf.random.set_seed(69)

    mem = []
    for o in range(len(train)):
        
        cluster = k_cluster[o]
        
        # create sequence for training and predicting 
        def generate_train_sequences(sequence, n_steps_in, n_steps_out):
            X, y = list(), list()
            for i in range(len(sequence)+n_steps_out):
                end_ix = i + n_steps_in
                out_end_ix = end_ix + n_steps_out
                if out_end_ix > len(sequence):
                    break
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
                X.append(seq_x)
                y.append(seq_y)
                input_seq = array(X)
                output_seq = array(y)
            return input_seq, output_seq

        # bring in appropriate data for this iteration
        train_seq = train[o] 

        # set parameters
        n_steps = 10
        n_features = 1
        epochs = 8
        batch_size = 32

        # Initialize
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)
        bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)
        
        # Build model, allow input of 'activation' to enable creating '2' models, one with tanh and another with relu
        # Eventually I'd like model creation to be much more in depth, adaptive, varying architecture, Bayesian search, etc
        # I also want to adjust the ART layer so it can pick out specifics but also generalities, 
        # e.g. cluster 1 is all numbers and cluster 2 is all letters: cluster 1.1 is '5' cluster 2.1 is 'B'
        def create_model_0(layers, activation):
            n_layers = len(layers)
            
            ## Encoder
            encoder_inputs = keras.layers.Input(shape=(None, 1))
            gru_cells = [keras.layers.GRUCell(hidden_dim, activation=activation, dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]

            encoder = keras.layers.RNN(gru_cells, return_sequences=True, return_state=True)
            encoder_outputs_and_states = encoder(encoder_inputs)
            encoder_outputs, state_h, state_h2, state_h3, state_c = encoder(encoder_inputs)
            encoder_states = encoder_outputs_and_states[1:]
            
            ## Decoder
            decoder_inputs = keras.layers.Input(shape=(None, 1))
            decoder_cells = [keras.layers.GRUCell(hidden_dim, activation=activation, dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]
            decoder_gru = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
            
            decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=encoder_states)
            [decoder_out, forward_h, forward_h2, forward_h3, forward_c] = decoder_gru(decoder_inputs, initial_state=encoder_states)
            
            decoder_dense1 = Dense(10, activation=activation, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
            decoder_outputs1 = decoder_dense1(decoder_out)
            decoder_dense2 = Dense(5, activation='relu', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
            decoder_outputs2 = decoder_dense2(decoder_out)
            
            merged = concatenate([decoder_outputs1, decoder_outputs2])
            
            decoder_dense = TimeDistributed(Dense(1, activation=activation))
            decoder_outputs = decoder_dense(merged)
            
            model_0 = keras.models.Model([encoder_inputs,decoder_inputs], decoder_outputs)
            return model_0
            
         
        neurons = 64
        batches = 1
        # test both models and assign id to them 
        # Only test on the first run or if it is a brand new cluster we haven't encountered yet 
        if o == 0 or cluster not in k_cluster[0:o]:
            fl = []
            for i in range(2):
            
                def run_model_0(model,batches,epochs,batch_size):
                    input_seq, output_seq = generate_train_sequences(train_seq, n_steps, 1)
                    encoder_input_data = input_seq
                    decoder_target_data = output_seq
                    decoder_input_data = np.zeros(decoder_target_data.shape)
                    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=.3,
                        shuffle=False)
                    total_loss.append(history.history['loss'])
                    total_val_loss.append(history.history['val_loss'])
                    return history, total_loss, total_val_loss 
                
                total_loss = []
                total_val_loss = []
                
                # create, compile and run models with different activations.  
                if i == 0:
                    activation = 'tanh'
                    exec_mod = i
                    model_0 = create_model_0([neurons,neurons,neurons,neurons], activation)
                    model_0.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')
                    history_0, total_loss_0, total_val_loss_0 = run_model_0(model_0,batches=batches, epochs=epochs,batch_size=batch_size)
                    total_loss_0 = [j for i in total_loss_0 for j in i]
                    total_val_loss_0 = [j for i in total_val_loss_0 for j in i]                    
                else:
                    activation = 'relu'
                    exec_mod = i
                    model_1 = create_model_0([neurons,neurons,neurons,neurons], activation)
                    model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')
                    history_1, total_loss_1, total_val_loss_1 = run_model_0(model_1,batches=batches, epochs=epochs,batch_size=batch_size)
                    total_loss_1 = [j for i in total_loss_1 for j in i]
                    total_val_loss_1 = [j for i in total_val_loss_1 for j in i]
                    
            # get final loss.  Will want to get more than just loss to determine model preference. To come ...
            final_loss_0 = total_val_loss_0[-1]
            final_loss_1 = total_val_loss_1[-1]

            # find winner, record, commit to 'memory'
            if final_loss_0 < final_loss_1:
                winner = 'model_0'
                winner_model = model_0
                model_0.save(directory + 'model_0' + str(cluster) + '.h5') # save model with cluster num assigned 
                cluster_model = cluster
            else:
                winner = 'model_1'
                winner_model = model_1
                model_1.save(directory + 'model_1' + str(cluster) + '.h5') # save model with cluster num assigned 
                cluster_model = cluster
        
        # If we have encountered this cluster, check memory ... and load model that won on similar data last time....
        else: 
            onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
            matchers = str(cluster) + '.'
            matching = [s for s in onlyfiles if matchers in s]
            matching = matching[0]
            winner = 'model_' + matching[7]
            winner_model = load_model(directory + 'model_' + matching[7] + str(cluster) + '.h5')
            cluster_model = cluster
        
        # winning models and their cluster 
        test = [winner, cluster_model]
        mem.append(test)
        mem_a = array(mem).reshape((-1))

        # make predictions 
        y_pred = []
        for i in range(len(xaxis) - n_steps):
            input_seq_test = train_seq[i:i+n_steps].reshape((1,n_steps,1))
            decoder_input_test = np.zeros((1,1,1))
            y = winner_model.predict([input_seq_test, decoder_input_test])
            y_pred.append(y)
            y_a = array(y_pred).reshape((-1))
            predictions = y_a.reshape((-1))
        
        plt.plot(predictions)
        plt.show()
    
