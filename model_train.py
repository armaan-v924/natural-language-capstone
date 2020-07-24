
from noggin import create_plot
import mygrad as mg
import numpy as np
from mynn.optimizers.SGD import SGD
from mappings import Mappings
import loss_and_acc_function as la
import text_embedding as te
import image_vector as iv
import nn_setup as nn
from gensim.models.keyedvectors import KeyedVectors

path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

plotter,fig,ax = create_plot(metrics=["loss","accuracy"])
model = nn.Model(512,50) 
optim = SGD(model.parameters, learning_rate=0.001)

plot_rate = 100
train_data, test_data = nn.sample_data(model)

for k in range(1000):
    

    wgood = te.text_embed(train_data[0],glove)
    wbad = te.text_embed(train_data[2],glove)
    good_caption = te.text_embed(train_data[1],glove)
    map = Mappings()
    bad_caption = te.text_embed(map.get_captions_imgID(wbad))
    dimg = iv.load_resnet()
    wimg = te.text_embed(dimg, glove)
    sgood = wimg * wgood
    sbad = wimg * wbad
    loss = la.loss(sgood,sbad)


    plotter.set_train_batch({"loss":loss.item()},batch_size=1, plot=False)
    if k % plot_rate== 0 and k > 0:
        plotter.set_train_epoch()
    
    loss.backward()
    optim.step()
    loss.null_gradients()
    
model.save_model("trained_parameters.npz")



