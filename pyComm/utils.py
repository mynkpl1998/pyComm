from __future__ import division, print_function
import torch
from numpy.random import randint

def symbol_error_rate(messages, messages_estimate):
    """
    Calculates and returns the symbol error rate.

    Parameters
    ----------
    messages           : 1d torch tensor of long
                            Original messages sent.
    
    messages_estimates : 1d torch tensor of long
                            Estimate of symbols sent.

    Raises
    ------
    TypeError:
                        If messages and messages_estimate are not 
                            torch tensors.
    ValueError:
                        If messages and messages_estimate are not
                            1d torch tensor of long
                        If the size messages and messages_estimate 
                        is not equal.
    
    Returns
    -------
    error_rate         : float, [0, 1]
                            Returns the error rate between 
                            messages and messages_estimate.
    """
    if type(messages) != torch.Tensor:
        raise TypeError("messages must be the 1d torch tensor of long. Got: %s"%(type(messages)))
    if len(messages.size()) != 1 or (messages.dtype != torch.int32 and messages.dtype != torch.int64):
        raise ValueError("messages must be the 1d torch tensor of long. Got size: %s, dtype: %s"%(messages.size(), messages.dtype))
    if type(messages_estimate) != torch.Tensor:
        raise TypeError("messages_estimate must be the 1d torch tensor of long. Got: %s"%(type(messages_estimate)))
    if len(messages_estimate.size()) != 1 or (messages_estimate.dtype != torch.int32 and messages_estimate.dtype != torch.int64):
        raise ValueError("messages_estimate must be the 1d torch tensor of long. Got size: %s, dtype: %s"%(messages_estimate.size(), messages_estimate.dtype))
    if messages.size(0) != messages_estimate.size(0):
        raise ValueError("messages and messages_estimate tensors should have same size. Got: messages_size :%d, messages_estimate_size : %d"%(messages.size(0), messages_estimate.size(0)))

    num_msgs = messages_estimate.size(0)
    error_rate = 1 - ((messages == messages_estimate).float().sum()/num_msgs)
    return error_rate.item()

if __name__ == "__main__":
    m = randint(0, 16, size=1024)
    m = torch.from_numpy(m)
    m_hat = randint(0, 16, size=1024)
    m_hat = torch.from_numpy(m_hat)
    print(symbol_error_rate(m, m_hat))