from __future__ import division, print_function
import torch
from numpy import log2, sqrt
from numpy.random import randint
from torch.distributions import Normal
from torch.nn import Embedding, Linear, Module
from torch.nn.functional import relu, normalize, softmax

class base_rx_tx(Module):
    """
    Base class to construct a neural network 
    based transmitter or receiver.

    Parameters
    ----------
    m             : integer
                    Constellation length.
    
    n             : integer
                    Number of channels to use.
    
    Attributes
    ----------
    constellation_length    : integer
                                Returns the constellation length.
    
    num_channel_uses        : integer
                                Returns the number of channel to use.
    
    layers:                 : list
                                Returns the list of parameterized layers
                                to build the tx/rx.
    
    Raises
    ------
    ValueError:
                    If the constellation length is not a power of 2.
                    If the number of channel is less than or equal to zero.
    
    """

    def __init__(self, m, n):
        super(base_rx_tx, self).__init__()
        num_bits_symb = log2(m)
        if num_bits_symb != int(num_bits_symb):
            raise ValueError("Constellation length must be a power of 2. Got constellation length :%d."%(m))
        self.__constellation_length = m 

        if n <= 0:
            raise ValueError("Number of channel uses must be greater than zero. Got channels to use :%d."%(n))
        self.__num_channel_uses = n
    
    @property
    def constellation_length(self):
        return self.__constellation_length
    
    @property
    def num_channel_uses(self):
        return self.__num_channel_uses
    
    @property
    def layers(self):
        raise NotImplementedError

class tx(base_rx_tx):
    
    """
    Construts a neural network based transmitter.

    Parameters
    ----------
    m             : integer
                    Constellation length.
    
    n             : integer
                    Number of channels to use.
    
    Attributes
    ----------
    constellation_length    : integer
                                Returns the constellation length.
    
    num_channel_uses        : integer
                                Returns the number of channel to use.
    
    layers                  : tuple
                                Returns parameterized layers of the transmitter.

    Raises
    ------
    ValueError:
                    If the constellation length is not a power of 2.
                    If the number of channel is less than or equal to zero.
    
    """
    def __init__(self, m, n):
        super().__init__(m, n)

        # Network layers
        self.__embded_layer = Embedding(num_embeddings=self.constellation_length, embedding_dim=256)
        self.__dense_1 = Linear(in_features=256, out_features=256)
        self.__dense_2 = Linear(in_features=256, out_features=2*self.num_channel_uses)
    
    @property
    def layers(self):
        return (self.__embded_layer,
                    self.__dense_1,
                    self.__dense_2)
    
    def forward(self, messages, train_mode=False, explore_variance=0.02):
        """
        Runs a forward pass on the messages and returns the
        constellation points.

        Parameters
        ----------
        messages    : 1d torch tensor of long.
                        Input symbols to be modulated.
        
        train_mode  : boolean
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
                    If messages is not a torch tensor.
                    If the any messages in torch tensor is greater than
                        or equal to m or negative integer.
                    If messages is not 1d torch tensor of long.
                    If explore_variance is less than zero.

        Returns
        -------
        baseband_signal     : 3d torch tensor of floats.
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
        if len(messages.size()) != 1 or (messages.dtype != torch.int32 and messages.dtype != torch.int64) :
            raise ValueError("message must have be 1d torch tensor of long. Got size: %s, dtype: %s"%(messages.size(), messages.dtype))
        if messages.max() >= self.constellation_length or messages.min() < 0:
            raise ValueError("messages must be >0 and <m. Got (%d, %d)"%(messages.min(), messages.max()))
        if explore_variance < 0:
            raise ValueError("Explore variance must be greater than zero., Got explore variance: %.4f."%(explore_variance))

        # Get batch size
        batch_size = messages.size(0)
        
        # Forward pass
        messages = messages.long()
        embed_out = relu(self.__embded_layer(messages))
        dense_1_out = relu(self.__dense_1(embed_out))
        dense_2_out = self.__dense_2(dense_1_out)
        
        # Reshape to complex
        complex_x = dense_2_out.view(batch_size, self.num_channel_uses, 2)
        
        # Normalize for unit energy
        normalized_x = normalize(complex_x, p=2, dim=2)

        # If training mode is enabled. Add noise to the signal for exploration.
        signal_dist = None
        if train_mode:
            explore_std_dev = torch.ones(normalized_x.size()) * sqrt(explore_variance) * 0.5
            signal_dist = Normal(sqrt(1 - explore_variance) * normalized_x, explore_std_dev)
        
        if train_mode:
            return normalized_x, signal_dist
        else:
            return normalized_x.detach()


class rx(base_rx_tx):
    
    """
    Construts a neural network based reciever.

    Parameters
    ----------
    m             : integer
                    Constellation length.
    
    n             : integer
                    Number of channels to use.
    
    Attributes
    ----------
    constellation_length    : integer
                                Returns the constellation length.
    
    num_channel_uses        : integer
                                Returns the number of channel to use.
    
    layers                  : tuple
                                Parameterized layers of the receiver.

    Raises
    ------
    ValueError:
                    If the constellation length is not a power of 2.
                    If the number of channel is less than or equal to zero.
    
    Returns
    -------
    symbol_logits       : 2d torch tensor of floats.
                            Logit value for each symbol
                            dim0: batch size
                            dim1: logits

    symbol_prob         : 2d torch tensor of floats.
                            Probablity for each symbol.
                            dim0: batch size
                            dim1: probabilty
    """

    def __init__(self, m, n):
        super().__init__(m, n)

        # Network layers
        self.__dense_1 = Linear(out_features=256, in_features=2*self.num_channel_uses)
        self.__dense_2 = Linear(out_features=256, in_features=256)
        self.__dense_3 = Linear(out_features=self.constellation_length, in_features=256)
    
    @property
    def layers(self):
        return (self.__dense_1,
                    self.__dense_2,
                    self.__dense_3)
    
    def forward(self, input_signal, train_mode=False):
        """
        Runs a forward pass on the input_siganl and returns the
        probablity over constellation points.

        Parameters
        ----------
        input_signal    : 3d torch tensor of floats.
                            Input symbols to be demodulated.
        
        train_mode      : boolean
                            Indicates training mode. Must be
                            set when receiver is in training
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

        # Get batch size
        batch_size = input_signal.size(0)

        # Reshape to real
        complex_2_real = input_signal.view(batch_size, -1)
        
        # Forward pass
        dense_1_out = relu(self.__dense_1(complex_2_real))
        dense_2_out = relu(self.__dense_2(dense_1_out))
        logits = self.__dense_3(dense_2_out)
        probs = softmax(logits, dim=1)

        if not train_mode:
            return logits.detach(), probs.detach()
        
        return logits, probs

if __name__ == "__main__":
    tx_model = tx(m=16, n=2)
    msgs = randint(0, tx_model.constellation_length, size=1024)
    msgs = torch.from_numpy(msgs).long()
    xp, dist = tx_model.forward(msgs, train_mode=True, explore_variance=0.02)
    
    rx_model = rx(m=16, n=2)
    logits, probs = rx_model.forward(input_signal=xp, train_mode=False)