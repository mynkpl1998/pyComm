from __future__ import division, print_function
import torch
from numpy import log2
from numpy.random import randint
from pyComm.models import base_rx_tx, tx, rx

class Modem:

    """
    Creates a End-to-End Modem comprising of a neural network
    based Tx and a Rx.

    Parameters
    ----------
    m             : integer
                    Constellation length.
    
    n             : integer
                    Number of channels to use.
    
    Attributes
    ----------
    constellation : 2d-ndarray of floats.
                    Modem constellation.
                    dim0: Message index.
                    dim1: Constellation point.

    constellation_length    : integer
                                Constellation length

    num_channel_uses        : integer
                                Number of channels to use.
    
    tx_model                : torch.nn.Module
                                Returns the torch tx 
                                model.
    
    rx_model                : torch.nn.Module
                                Returns the torch rx
                                model.

    Raises
    ------
    ValueError
                    If the constellation length is not a power of 2.
                    If the number of channel is less than or equal to zero.
    """

    def __init__(self, m, n):
        num_bits_symb = log2(m)
        if num_bits_symb != int(num_bits_symb):
            raise ValueError("Constellation length must be a power of 2. Got constellation length :%d."%(m))
        self.__constellation_length = m

        if n <= 0:
            raise ValueError("Number of channel uses must be greater than zero. Got channels to use :%d."%(n))
        self.__num_channel_uses = n

        self.__tx_model = tx(m, n)
        self.__rx_model = rx(m, n)
    
    @property
    def constellation(self):
        raise NotImplementedError

    @property
    def constellation_length(self):
        return self.__constellation_length
    
    @property
    def tx_model(self):
        return self.__tx_model
    
    @property
    def rx_model(self):
        return self.__rx_model
    
    @property
    def num_channel_uses(self):
        return self.__num_channel_uses
    
    def modulate(self, messages, train_mode=False, explore_variance=0.02):
        """
        Modulates the array of integer(symbols) to constellation symbols.

        Parameters
        ----------
        messages            : 1d torch tensor of long.
                                Input symbols to be modulated.
        
        train_mode          : boolean
                                Indicates training mode. Must be
                                set when trasmitter is in training
                                mode.
                
        explore_variance    : float
                                Exploration noise variance.
                                Used only in training when
                                train mode is set to True.
        
        Raises
        ------
        ValueError:
                    If message is not a torch tensor of long.
                    If the any messages in torch tensor is greater than
                        or equal to m or negative integer.

        Returns
        -------
        baseband_symbols    : 3d torch tensor of floats.
                                Modulated complex symbol.
                                dim0: Batch size
                                dim1: Number of channels to use.
                                dim2: Complex dim. Must be always 2. (Real and Img)
        
        baseband_dist       : torch Normal dist object.
                                Returns distribution from which signal
                                was sampled. This is returned only during
                                train mode.
        """
        if type(messages) != torch.Tensor:
            raise TypeError("message must be an 1d torch tensor of long. Got: %s"%(type(messages)))
        if len(messages.size()) != 1 or (messages.dtype != torch.int32 and messages.dtype != torch.int64):
            raise ValueError("message must have be 1d torch tensor of long. Got size: %s, dtype: %s"%(messages.size(), messages.dtype))
        if messages.max() >= self.constellation_length or messages.min() < 0:
            raise ValueError("messages must be >0 and <m. Got (%d, %d)"%(messages.min(), messages.max()))
        
        x = self.__tx_model.forward(messages=messages, train_mode=train_mode, explore_variance=explore_variance)
    
        if not train_mode:
            x = x.detach()
            return x
        return x[0], x[1]
    
    def demodulate(self, input_signal, train_mode=False):
        """
        Demodulates the ndarray of complex signals to symbols.

        Parameters
        ----------
        input_signal        : 3d torch tensor of long.
                                Input symbols to be demodulated.
        
        train_mode          : boolean
                                Indicates training mode. Must be
                                set when reciever is in training
                                mode.
        
        Raises
        ------
        ValueError:
                            If input signal is not a 3d torch tensor of floats.
                            If size of dim2 (last) dim of input_signal is not 2.

        Returns
        -------
        demodulated_symbols    : 1d torch tensor of long.
                                    Demodulated symbols.
        
        symbol_logits          : 2d torch tensor of floats.
                                    Logit value for each symbol
                                    dim0: batch size
                                    dim1: logits

        symbol_prob            : 2d torch tensor of floats.
                                    Probablity of each symbol.
                                    dim0: batch size
                                    dim1: probabilty
        """
        if type(input_signal) != torch.Tensor:
            raise TypeError("input_signal must be a 3d torch tensor of floats. Got: %s"%(type(input_signal)))
        if len(input_signal.size()) != 3 or (input_signal.dtype != torch.float32 and input_signal.dtype != torch.float64):
            raise ValueError("input_signal must be a 3d torch tensor of floats. Got size: %s, dtype: %s"%(input_signal.size(), input_signal.dtype))
        if input_signal.size(2) != 2:
            raise ValueError("Dim 2 of input_signal, must have size 2. got dim2: %d"%(input_signal.size(2)))
        
        logits, probs = self.__rx_model.forward(input_signal=input_signal,
                                                train_mode=train_mode)
        if not train_mode:
            logits = logits.detach()
            probs = probs.detach()
        
        # Symbol decoding
        _, demodulated_symbols = probs.max(dim=1)
        return demodulated_symbols.long(), logits, probs
        
    
if __name__ == "__main__":
    mod = Modem(m=16, n=4)
    messages = randint(0, mod.constellation_length, size=1024)
    messages = torch.from_numpy(messages).long()
    x = mod.modulate(messages)
    
    m_hat = mod.demodulate(x, train_mode=False)
    print(m_hat)