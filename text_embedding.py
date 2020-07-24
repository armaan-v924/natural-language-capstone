from nltk.tokenize import word_tokenize
import string
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from mappings import Mappings
from collections import Counter

def text_embed(text, glove, mappings):
    """
    generates embedding for the text
    
    Parameters
    ----------
    text: string - text to be embedded
    
    Returns
    ----------
    embedded_text: np.array(50,)
    """

    #remove punctuation, all lowercase, split by space
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, "")
    tokens = word_tokenize(text)
    
    #for IDFs - set up counter for words across all documents
    
    all_captions = mappings.captions
    N = len(all_captions)
    
    #join all captions, lowercase, no punctuation, split by space
    all_captions_tokens = " "
    all_captions_tokens = all_captions_tokens.join(all_captions)
    all_captions_tokens = all_captions_tokens.lower()
    for p in string.punctuation:
        all_captions_tokens = all_captions_tokens.replace(p, "")
    all_captions_tokens = word_tokenize(all_captions_tokens)
    
    #set up counter
    c = Counter(all_captions_tokens)
    
    #generate Glove-50 embeddings for all words in text, shape: (len(tokens), 50)
    embedded_text = np.zeros((len(tokens), 50))
    for word_idx in range(len(tokens)):
        nt = c[tokens[word_idx]] #num times word appears across all documents
        idf = np.log10(N / nt)
        embedded_text[word_idx] = idf * glove[tokens[word_idx]]
    
    #sum embeddings across all words in the text, shape: (50,)
    embedded_text = np.sum(embedded_text, axis=0)
    
    return embedded_text


"""
Run this once to load the glove 50 set *also will need to change path later*, pass into text_embed to use
"""
#path = r"./glove.6B.50d.txt.w2v"
#glove = KeyedVectors.load_word2vec_format(path, binary=False)


