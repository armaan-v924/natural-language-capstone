
from noggin import create_plot
import mygrad as mg
import numpy as np
from mynn.optimizers.sgd import SGD
from mappings import Mappings
import loss_and_acc_function as la
import text_embedding as te
import image_vector as iv
import nn_setup as nn
from gensim.models.keyedvectors import KeyedVectors
import pickle

path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

plotter,fig,ax = create_plot(metrics=["loss","accuracy"])
model = nn.Model(512,50) 
optim = SGD(model.parameters, learning_rate=0.001)

plot_rate = 100
map = Mappings()
idfs = te.inverse_document_frequency(map.captions)
caption_tokens = te.get_all_captions_tokens(map.captions)
text_embeds = {}
for cap_id, cap in zip(map.captionID, map.captions):
    text_embeds[cap_id] = te.text_embed(cap, glove, map.captions, caption_tokens, idfs)

batch_size = 32
resnet = iv.load_resnet()

train_data, test_data = nn.sample_data(map, resnet, glove, text_embeds)
pickle.dump((train_data,test_data), open( "train_test_data.p", "wb" ) )

for epoch_rate in range(5):
    print(epoch_rate)
    idxs = np.arange(len(train_data))
    np.random.shuffle(idxs)
    for batch_rate in range(len(train_data) // batch_size):
        # print(batch_rate,len(train_data)//batch_size)
        batch_indices = idxs[batch_rate * batch_size:(batch_rate + 1) * batch_size]
        batch = train_data[batch_indices]
        dgood = []
        dbad = []
        wcaption = []
        for x in batch:
            dgoodtemp = iv.get_resnet_vector(x[0], resnet=resnet)
            dbadtemp = iv.get_resnet_vector(x[2], resnet=resnet)
            wcaptiontemp = text_embeds[x[1]]

            dgood.append(dgoodtemp)
            dbad.append(dbadtemp)
            wcaption.append(wcaptiontemp)
        wcaption = np.array(wcaption)
        wgood = model(np.array(dgood))
        wbad = model(np.array(dbad))
        sgood = np.squeeze(np.expand_dims(wcaption, axis=1) @ np.expand_dims(wgood, axis=2)).astype(float)
        sbad = np.squeeze(np.expand_dims(wcaption, axis=1) @ np.expand_dims(wbad, axis=2)).astype(float)
        loss = la.loss(sgood, sbad)
        accuracy = la.acc(sgood, sbad)
        loss.backward()
        optim.step()
        loss.null_gradients()

        plotter.set_train_batch({"loss": loss.item(), "accuracy": accuracy}, batch_size=batch_size)

    if epoch_rate % plot_rate == 0 and epoch_rate > 0:
        plotter.set_train_epoch()
        plotter.set_test_epoch()

model.save_model("trained_parameters.npz")



