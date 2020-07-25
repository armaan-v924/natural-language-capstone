import pickle

def load_resnet(resnet_path="resnet18_features.pkl"):
    """
    Has to be run at least once before calling get_resnet_vector
    
    Parameters
    ----------
    resnet_path : String (optional), path to resnet18_features.pkl
                   if none, just assumes resnet18_features.pkl is
                   is in the same folder and calls the file
    
    Returns
    -------
    database: dictionary mapping image IDs to image feature vectors
    
    """
    with open(resnet_path, mode="rb") as opened_file:
        resnet = pickle.load(opened_file)
    
    return resnet

def get_resnet_vector(image_id, resnet=None):
    """
    Return resnet18 feature vector for image_id    

    Parameters
    ----------
    image_id : int, ID# pointing to a COCO dataset image
    resnet : dict (optional), loaded resnet dictionary
             If none, just assumes resnet18_features.pkl is
             in the same folder and loads the database
    
    Returns
    -------
    np.ndarray: resnet18 feature vector for image_id
    
    """

    if resnet is None:
        resnet = load_resnet()

    try:
        return resnet[image_id].flatten()
    except KeyError:
        return 0