from mynn.layers.dense import dense

from mygrad.nnet.initializers import glorot_normal, constant

import numpy as np

class Model():
    def __init__(self, dim_input, dim_output):
        """This initializes the layer for our Linear Encoder, and sets it
        as an attribute of the model.
        
        Parameters
        ----------
        dim_input : int
            The size of our input
            
        dim_output : int
            The size of the output layer 
        """
        self.layer = dense(dim_input, dim_output)

    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
                
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, dim_input)
            A batch of data consisting of M pieces of data,
            each with a dimentionality of dim_input.
            
        Returns
        -------
        mygrad.Tensor, shape=(M, num_out)
            The model's imbedding for each image.
        '''
        return self.layer(x)

    @property
    def parameters(self):
        """
        Returns
        -------
        mygrad.Tensor"""
        return self.layer.parameters

def sample_data(full_dataset):
    # IGNORE THIS IT"S NOT DONE YET SORRY
    all_cap = full_dataset.get_all_caption_IDs() # convenience function
    all_cap = np.repeat(all_cap, 10)
    total_cap_number = len(all_cap)
    train_cap = all_cap[:total_cap_number*(4/5)]
    test_cap = all_cap[-total_cap_number*(1/5):]

