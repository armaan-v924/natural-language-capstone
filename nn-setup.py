from mynn.layers.dense import dense

from mygrad.nnet.initializers import glorot_normal, constant

import numpy as np
import mappings

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
    '''Creates training set and testing set given the full class of data/Mappings
            
    Parameters
    ----------
    full_dataset : Mappings,
                   An instance of Mappings that correlates to the dataset that will
                   be used as training and testing data
        
    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
    Training data (80% of given dataset), Testing data (20% of given dataset)
    Each np.ndarray is shape (num_datapoints, 3) where the 3 columns are good image IDs,
    good caption IDs, bad caption IDs
    '''
    all_cap = full_dataset.all_captionID()    
    all_bad = np.array([])
    all_img = np.array([])
    total_cap = len(all_cap)    

    for i in range(0, total_cap):
        possible = np.random.randint(0, total_cap, size=(25,))
        good_img = full_dataset.get_imageID(all_cap[i*10])
        good_w = full_dataset.get_cap_vector(all_cap[i*10])
        
        diff = np.array([])
        for p in possible:
            bad_img = full_dataset.get_imageID(p)
            if bad_img != good_img:
                bad_w = full_dataset.get_cap_vector(p)
                diff = np.append(diff, np.dot(bad_w, good_w))
            else:
                diff = np.append(diff, 0)
                
        diff = np.argsort(diff)
        worst = possible[diff[-10:]]

        all_bad = np.append(all_bad, worst)
        all_img = np.append(all_img, good_img)
     
    all_cap = np.repeat(all_cap, 10)
    all_img = np.repeat(all_img, 10)
    test_num = int(len(all_cap)*(1/5))

    train_cap = all_cap[:test_num*4]
    test_cap = all_cap[-test_num:]
    
    train_bad = all_bad[:test_num*4]
    test_bad = all_bad[-test_num:]

    train_img = all_img[:test_num*4]
    test_img = all_img[-test_num:]
    
    train_data = np.array(list(zip(train_img, train_cap, train_bad)))
    test_data = np.array(list(zip(test_img, test_cap, test_bad)))
    
    return train_data, test_data