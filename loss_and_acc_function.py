#loss and accuracy function
import mygrad as mg
import numpy as np
def loss(sgood, sbad):
    """Calcuates the loss of sgood and sbad using a margin
ranking loss.

    Parameters
    ----------
        sgood (mg.Tensor): Wimg * Wgood
        sbad (mg.Tensor): Wimg * Wbad

    Returns
    ----------
        loss: max(0, margin-(sgood-sbad))
    """
    margin = 0.1
    return mg.nnet.losses.margin_ranking_loss(sgood, sbad, 1, margin, constant=False)

def acc(prediction,truth):
    """Calculates the accuracy of the predicted models over the truth.

    Parameters
    ----------
        prediction: TBA
        truth: TBA

    Returns
    ---------
        mean
    """
    counter1 = 0
    for i in range(len(prediction)):
        counter1+=int((prediction[i] >= truth[i]).all())
    return counter1/len(prediction)
