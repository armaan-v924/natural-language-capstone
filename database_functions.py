import pickle
# import trained model
import image_vector as iv

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

def add_image(image_id):
    """
    adds a new image to the database {image_vector: semantic_embeddings}
    
    Parameters
    ----------
    profile: Profile of the person to add
    
    """
    database = load_db()
    image_vector = iv.get_resnet_vector(image_id)
    semantic_embeddings = get_embeddings(image_id) #after training
    database[tuple(image_vector)] = (image_id, semantic_embeddings)
    save_db(database)