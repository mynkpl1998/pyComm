from __future__ import print_function, division
import os
import torch
from numpy import log2
from torch.optim import Adam
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

    num_channel_uses        : integer
                                Number of channels to use.

    rx_loss_fn              : torch.nn.loss
                                Returns the rx loss function.

    tx_loss_fn              : torch.nn.loss
                                Returns the tx loss function.
    
    available_optimizers    : list
                                Returns the list of supported 
                                optimizers.
    
    channel                 : pyComm.AwgnChannel
                                Returns the awgn channel object.
    
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
        self.__tx_loss_fn = self._tx_loss_fn
        self.__supported_optimizers = ['adam']

    def _tx_loss_fn(self, logits, messages, log_probs):
        """
        Defines the cross entropy loss or reward
        used by tx to train the model.
        
        Paramters
        ---------
        * logits           : torch 1d tensors of long.
                            The prediction made by tx model.
        
        * messages         : torch 1d tensor of long.
                            Actual transmitted messages.
        
        * log_probs         : torch 3d tensor of floats.
                            Log probs of the transmitted signals.
                            dim 0: Batch size.
                            dim 1: Number of channel uses.
                            dim 2: Complex dim. Must be always 2. (Real and Img)

        Returns
        -------
        loss_value          : float
                                Returns the loss value.
        """
        per_example_loss = CrossEntropyLoss(reduction='none')(logits, messages)
        batch_size = log_probs.size(0)
        loss = log_probs.view(batch_size, -1).sum(dim=1).mul(per_example_loss).sum()
        loss /= batch_size
        return loss
    
    @property
    def rx_loss_fn(self):
        return self.__rx_loss_fn
    
    @property
    def available_optimizers(self):
        return self.__supported_optimizers

    @property
    def tx_loss_fn(self):
        return self.__tx_loss_fn

    @property
    def modem(self):
        return self.__modem
    
    @property
    def constellation_length(self):
        return self.__constellation_length
    
    @property
    def num_channel_uses(self):
        return self.__num_channel_uses
    
    @property
    def channel(self):
        return self.__channel
    
    def rx_train(self, batch_size, snr_db, optimizer):
        """
        Run one iteration of learning step on the rx model and
        returns the mean loss on the complete batch.

        Parameters
        ----------
        batch_size          : integer
                                Batch size to use for rx training.
        
        snr_db              : float
                                snr to use during training.
        
        optimizer           : torch.optim
                                torch optimizer to use for rx
                                training.
    
        Raises
        ------
        ValueError:
                            If batch size less one.
                            If batch size is not an integer.

        Returns
        -------
        loss_value:         : float
                                Returns the mean loss on 
                                the complete batch.

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

        # SGD update on rx parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def tx_train(self,
                 batch_size,
                 explore_variance,
                 snr_db,
                 optimizer):
        """
        Run one iteration of learning step on the rx model and
        returns the mean loss on the complete batch.

        Parameters
        ----------
        batch_size          : integer
                                Batch size to use for rx training.
        
        explore_variance    : float
                                tx exploration variance.
        
        snr_db              : float
                                snr to use during training.
        
        optimizer           : torch.optim
                                torch optimizer to use for tx
                                training.
    
        Raises
        ------
        ValueError:
                            If batch size less one.
                            If batch size is not an integer.
                            If explore_variance is not a float.

        Returns
        -------
        loss_value:         : float
                                Returns the mean loss on 
                                the complete batch.

        """
        if type(batch_size) != int:
            raise ValueError("batch_size should be an integer. Got: %s"%(type(batch_size)))
        if batch_size < 1:
            raise ValueError("batch_size should be greater then zero. Got batch_size :%d"%(batch_size))
        if explore_variance != float(explore_variance):
            raise ValueError("explore variance must be an float. Got: %s"%(type(explore_variance)))

        # Generate random set of messages to transmit.
        messages = torch.randint(0, self.constellation_length, size=(batch_size, )).long()
        
        # Modulate signals
        x_mean, signal_dist = self.modem.modulate(messages=messages,
                                                  train_mode=True,
                                                  explore_variance=explore_variance)
        x = signal_dist.sample()

        # Channel noise
        y = self.channel.propagate(input_signal=x,
                                   snr_db=snr_db)
        
        # Demodulate complex signals
        _, logits, __ = self.modem.demodulate(input_signal=y,
                                               train_mode=False)
        
        # Calculate per example loss
        loss = self.tx_loss_fn(logits=logits,
                               messages=messages,
                               log_probs=signal_dist.log_prob(x))
        
        # SGD update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def calculate_error_rate(self, snr_db, sample_size=10000):
        """
        Calculates the monte carlo estimate of the 
        error rate of the modulation at given SNR.

        Parameters
        ----------

        snr_db                      : float
                                        Signal to noise ratio in dB.

        sample_size                 : integer
                                        Number of messages to use
                                        to estimate the error rate.
                                        Default: 10^3.
        

        Returns
        ------
        error_rate:                 : float
                                        Return the error rate of the
                                        modulation.         

        Raises
        ------
        ValueError:
                                    If the sample_size is not greater than zero.
                                    If sample_size is not an integer.
                                    If snr_db is not a valid float

        """
        if sample_size != int(sample_size):
            raise ValueError("sample_size must be an integer. Got: %s"%(type(sample_size)))
        sample_size = int(sample_size)
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero. Got: %d"%(sample_size))
        if snr_db != float(snr_db):
            raise ValueError("snr_db must be a float. Got :%s"%(type(float)))
        snr_db = float(snr_db)
        
        # Generate random set of messages to transmit.
        messages = torch.randint(0, self.constellation_length, size=(sample_size, )).long()
    
        # Modulate symbols
        x = self.modem.modulate(messages=messages,
                                train_mode=False)
        
        # Channel noise
        y = self.__channel.propagate(input_signal=x, snr_db=snr_db)
        
        # Demodulate complex signal
        messages_pred, _, __ = self.modem.demodulate(input_signal=y, train_mode=False)

        # Calculate error rate
        err_rate = 1 - ((messages_pred == messages).float().sum()/sample_size)

        return err_rate.item()

    def train(self,
                batch_size, 
                snr_db,
                epochs,
                explore_variance,
                tx_optimizer_config_dict,
                rx_optimizer_config_dict,
                tx_optimizer='adam',
                rx_optimizer='adam',
                verbose=True,
                sample_size=10**3):
        """
        Run one iteration of learning step on the rx and tx model.

        Parameters
        ----------
        batch_size          : integer
                                Batch size to use for training.
        
        snr_db              : float
                                snr to use during training.
        
        epochs              : integer
                                training epochs.
        
        explore_variance    : float
                                tx exploration variance.
        
        rx_optimizer        : str
                                Must belong to supported list of
                                optimizers.
                                torch optimizer to use for rx
                                training. Default: 'adam'.
        
        tx_optimizer        : str
                                Must belong to supported list of
                                optimizers.
                                torch optimizer to use for tx
                                training. Default: 'adam'.
        
        verbose             : boolean
                                Enable/disable prints during 
                                training.
        
        sample_size         : integer
                                Number of messages to use
                                to estimate the error rate.
                                Default: 10^3.

        
        Raises
        ------
        ValueError:
                            If batch size less one.
                            If batch size is not an integer.
                            If optimizer is not a valid torch
                            optimizer.
                            If training epoch is not greater than
                            one.
                            If verbose is not a boolean.

        Returns
        -------

        """
        if type(batch_size) != int:
            raise ValueError("batch_size should be an integer. Got: %s"%(type(batch_size)))
        if batch_size < 1:
            raise ValueError("batch_size should be greater then zero. Got batch_size :%d"%(batch_size))
        if tx_optimizer != str(tx_optimizer) or rx_optimizer != str(rx_optimizer):
            raise ValueError("tx or rx optimizer must be an str. Got, tx optimizer: %s, rx optimizer: %s"%(type(tx_optimizer), type(rx_optimizer)))
        if tx_optimizer not in self.available_optimizers:
            raise ValueError("%s optimizer is not supported yet. Supported optimizers : %s"%(tx_optimizer, self.available_optimizers))
        if rx_optimizer not in self.available_optimizers:
            raise ValueError("%s optimizer is not supported yet. Supported optimizers : %s"%(rx_optimizer, self.available_optimizers))
        if epochs != int(epochs):
            raise ValueError("epochs must be an integer. Got: %s"%(type(epochs)))
        epochs = int(epochs)
        if epochs <=0 :
            raise ValueError("epoch muster > 0., Got, epochs: %d"%(epochs))
        if verbose != bool(verbose):
            raise ValueError("verbose must be a boolean. Got: %s"%(type(verbose)))
        
        # Optimizers
        tx_optimizer = Adam(self.modem.tx_model.parameters())
        rx_optimizer = Adam(self.modem.rx_model.parameters())

        # logging
        log_dict = {
            'rx_loss': [],
            'tx_loss': [],
            'request_iters': epochs,
            'completed_iters': 0,
            'error_rate': []
        }

        # train tx and rx for given number of epochs
        for _iter_ in range(0, epochs):
            
            # Alternate training regime
            rx_loss = self.rx_train(batch_size=batch_size, snr_db=snr_db, optimizer=rx_optimizer)
            tx_loss = self.tx_train(batch_size=batch_size, explore_variance=explore_variance, optimizer=tx_optimizer, snr_db=snr_db)

            # logging
            log_dict['rx_loss'].append(rx_loss)
            log_dict['tx_loss'].append(tx_loss)
            
            # calculate error rate
            err_rate = self.calculate_error_rate(snr_db=snr_db)
            log_dict['error_rate'].append(err_rate)

            if verbose:
                if os.name == 'nt': # Windows system
                    os.system('cls')
                else:
                    os.system('clear')
                
                print("Training Stats:\n\t1. Iterations Completed: %d/%d. \n\t2. RX loss: %.4f\n\t2. TX Loss: %.4f\n\t4. Error Rate (SNR=%.2f dB): %.4f"%(_iter_+1, 
                                                                                                                                            epochs,
                                                                                                                                            rx_loss,
                                                                                                                                            tx_loss,
                                                                                                                                            snr_db,
                                                                                                                                            err_rate))

        return log_dict