import json
import matplotlib.pyplot as plt
from mappings import Mappings
import numpy as np


def cos_distance(d1, d2):
    '''
    Computes the cosine distance between face descriptors

    Parameters:
    -----------
    d1: facial descriptor for first image, numpy array
    d2: facial descriptor for second image, numpy array

    Returns:
    --------
    A number from [0,2] representing the cosine distance of d1 and d2
    '''

    return 1 - np.dot(d2, d1) / (np.linalg.norm(d1) * np.linalg.norm(d2))

def find_topk_images(k, embed_query, database):
    '''
    Finds the top k images based on the embed_query of the caption using database

    Parameters:
    -----------
    k: number of top images to return
    embed_query: the embed caption based on text/caption
    database: a dictionary that maps the image feature vectors to the semantic embeddings

    Returns:
    --------
    returns image ids corresponding to the top k images in database
    '''
    distances = []
    for feature, embedded in database.items():
        distances.append((cos_distance(embed_query,embedded[1]), feature, embedded[0]))
    top_images = sorted(distances, reverse=True)[:k]
    return [image[2] for image in top_images]

def display_images(image_ids):
    '''
    Displays the top images based on image ids

    Parameters:
    -----------
    image_ids: a list of ids for each image

    Returns:
    --------
    None; but displays images of closest matches to caption
    '''
    maps = Mappings()
    fig, ax = plt.subplot(len(image_ids),1)
    for i in range(len(image_ids)):
        img_url = maps.imgURL[image_ids[i]]
        a = plt.imread(img_url)
        ax[i].imshow(a)
    ax.show()

