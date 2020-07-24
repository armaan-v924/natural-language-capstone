import database_functions
import text_embedding
import findImages
from gensim.models.keyedvectors import KeyedVectors
from mappings import Mappings

#load glove-50 --> will need to change path
#if you need to access glove, you should do it from here so it only has to load once
path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

print("Developed by @therealshazam\n")
print("What would you like to do?")
function = input("1. Update the database\n2. Find an image via caption\n")
if function == '1':
    mapping = Mappings()
    ids = mapping.captions()
    database_functions.add_images(ids, "trained_parameters.npz")
elif function == '2':
    #ask for image caption and embed
    caption = input("Please enter a caption for the image: ")
    embedded_caption = text_embedding.text_embed(caption, glove)

    #dot product with database to find similarity scores
    database = database_functions.load_db()

    #find top k scores (for n images to display) and their images
    k = 4
    image_ids = findImages.find_topk_images(k, embedded_caption, database)

    #display images
    findImages.display_images(image_ids)
else:
    print("Sorry, something went wrong.")

