import pickle
# import trained model
# import image feature vectors thing

def load_db(pathname):
    """
    returns the stored database from a pickle file
    
    Parameters
    ----------
    pathname: string
    
    Returns
    -------
    database: dictionary mapping names to profiles
    
    """
    with open(pathname, mode="rb") as opened_file:
        database = pickle.load(opened_file)
    return database
    
def save_db(database, pathname):
    """
    saves the given database into a pickle file
    
    Parameters
    ----------
    database: dictionary
    pathname: string
    
    """
    with open(pathname, mode="wb") as opened_file:
        pickle.dump(database, opened_file)

def add_image(image_id):
    """
    adds a new image to the database {image_vector: semantic_embeddings}
    
    Parameters
    ----------
    profile: Profile of the person to add
    
    """
    database = load_db("database.pkl")
    image_vector = get_vector(image_id)
    semantic_embeddings = get_embeddings(image_id)
    database[image_vector] = (image_id, semantic_embeddings)
    save_db(database, "database.pkl")
    
def remove_image(image_id):
    """
    removes a profile from the database
    
    Parameters
    ----------
    profile: Profile of the person to remove
    
    """
    database = load_db("database.pkl")
    database.pop(profile.name)
    save_db(database, "database.pkl")