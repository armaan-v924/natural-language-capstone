import pickle
# import trained model
import image_vector as iv
import numpy as np
from nn_setup import Model
import time

def load_db():
    """
    returns the stored database from a pickle file
    
    Parameters
    ----------
    pathname: string
    
    Returns
    -------
    database: dictionary mapping names to profiles
    
    """
    with open("database.p", mode="rb") as opened_file:
        database = pickle.load(opened_file)
    return database
    
def save_db(database):
    """
    saves the given database into a pickle file
    
    Parameters
    ----------
    database: dictionary
    pathname: string
    
    """
    with open("database.p", mode="wb") as opened_file:
        pickle.dump(database, opened_file)

# def add_image(image_id):
#     """
#     adds a new image to the database {image_vector: semantic_embeddings}
    
#     Parameters
#     ----------
#     profile: Profile of the person to add
    
#     """
#     database = load_db()
#     image_vector = iv.get_resnet_vector(image_id)
#     semantic_embeddings = get_embeddings(image_id) #after training
#     database[tuple(image_vector)] = (image_id, semantic_embeddings)
#     save_db(database)

def add_images(image_ids, param_path):
    model = Model(512,50)
    model.load_model(param_path)
    database = load_db()
    resnet = iv.load_resnet()
    keys = []
    ivs = []
    new_ids = []
    for i in range(len(image_ids)):
        print(i)
        res = iv.get_resnet_vector(image_ids[i],resnet = resnet)
        if type(res) is np.ndarray:
            keys.append(tuple(res))
            ivs.append(res)
            new_ids.append(image_ids[i])
    ivs = np.array(ivs)
    new_ids = np.array(new_ids)
    semantic_embeddings = model(ivs).data

    val = list(zip(new_ids, semantic_embeddings))
    add = dict(tuple(zip(keys, val)))
    database.update(add)
    save_db(database)