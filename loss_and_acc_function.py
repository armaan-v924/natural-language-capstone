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
    return mg.max(0,margin-(sgood-sbad))
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
    return mg.mean(np.argmax(prediction,axis=1) == truth)
