import database_functions
import text_embedding
import findImages

#load glove-50 --> will need to change path
#if you need to access glove, you should do it from here so it only has to load once
path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

#ask for image caption and embed
caption = input("Please enter a caption for the image: ")
embedded_caption = text_embedding.text_embed(caption, glove)

#dot product with database to find similarity scores
database = database_functions.load_db()
similarity_scores = embedded_caption @ database

#find top k scores (for n images to display) and their images
k = 4
image_ids = findImages.find_topk_images(k, embedded_caption, database)

#display images
findImages.display_images(image_ids)

