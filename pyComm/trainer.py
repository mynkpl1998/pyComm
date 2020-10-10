from __future__ import print_function, division
import torch
from numpy import log2
from pyComm.modulation import Modem
from torch.nn import CrossEntropyLoss
from pyComm.channels import AwgnChannel

class alterateTrainer(object):
    """
    Constructs a trainer object for end-to-end communication
    systerm and uses alterate training region to adjust
    the weights of transmitter and receiver.

    Parameters
    ----------
    m             : integer
                    Constellation length.
    
    n             : integer
                    Number of channels to use.
    
    Attributes
    ----------
    modem         : pyComm.modulations.Modem
                    Returns the Modem object.
    
    constellation_length    : integer
                                Constellation length

    nnum_channel_uses       : integer
                                Number of channels to use.
    
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

        self.__modem = Modem(m, n)
        self.__channel = AwgnChannel()
        self.__rx_loss_fn = CrossEntropyLoss(reduction='mean')
    
    def @property(self):
        return self.__rx_loss_fn

    @property
    def modem(self):
        return self.__modem
    
    @property
    def constellation_length(self):
        return self.__constellation_length
    
    @property
    def num_channel_uses(self):
        return self.__num_channel_uses
    
    def tx_train(self, batch_size, explore_variance):
        print("tx train")
    
    def rx_train(self, batch_size, snr_db):
        """
        Run one iteration of learning step on the rx model.

        Parameters
        ----------
        batch_size          : integer
                                Batch size to use for rx training.
        
        snr_db              : float
                                snr to use during training.
        
        Raises
        ------
        ValueError:
                            If batch size less one.
                            If batch size is not an integer.

        Returns
        -------

        """
        if type(batch_size) != int:
            raise ValueError("batch_size should be an integer. Got: %s"%(type(batch_size)))
        if batch_size < 1:
            raise ValueError("batch_size should be greater then zero. Got batch_size :%d"%(batch_size))
        
        # Generate random set of messages to transmit.
        messages = torch.randint(0, self.constellation_length, size=(batch_size, )).long()
        
        # Modulate symbols
        x = self.modem.modulate(messages=messages,
                                train_mode=False)
        
        # Channel noise
        y = self.__channel.propagate(input_signal=x, snr_db=snr_db)
        
        # Demodulate complex signal
        _, logits, __ = self.modem.demodulate(input_signal=y, train_mode=True)
        
        # log likelihood loss
        loss = self.__rx_loss_fn(logits, messages)
        

    
    def train(self, batch_size, snr_db):
        """
        Run one iteration of learning step on the rx and tx model.

        Parameters
        ----------
        batch_size          : integer
                                Batch size to use for training.
        
        snr_db              : float
                                snr to use during training.
        
        Raises
        ------
        ValueError:
                            If batch size less one.
                            If batch size is not an integer.

        Returns
        -------

        """
        if type(batch_size) != int:
            raise ValueError("batch_size should be an integer. Got: %s"%(type(batch_size)))
        if batch_size < 1:
            raise ValueError("batch_size should be greater then zero. Got batch_size :%d"%(batch_size))
    
        self.rx_train(batch_size=batch_size, snr_db=snr_db)