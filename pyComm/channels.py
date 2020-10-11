from __future__ import division, print_function
import torch

__all__ = ["AwgnChannel"]

class AwgnChannel(object):

    """
    Constructs an Additive White Gaussian Noise(AWGN) Channel.
    
    """ 

    def __init__(self):
        pass

    def propagate(self, input_signal, snr_db):

        """
        Mesaures the energy of the transmit signal and adds the
        gaussian noise to it according to the signal-to-noise
        ratio.

        Parameters
        ----------
        input_signal: 3D torch tensor of floats.
            Complex input signal to the channel.
            dim 0: Batch size.
            dim 1: Number of channel uses.
            dim 2: Complex dim. Must be always 2. (Real and Img)
        
        snr_db: float
            Signal to noise ratio on dB.

        Returns
        ------
        output_signal: 3D torch tensor of floats.
            Output signal with gaussian noise.
            dim 0: Batch size.
            dim 1: Number of channel uses.
            dim 2: Complex dim. Must be always 2. (Real and Img)
        
        Raises
        ------
        ValueError:
                    If message is not a torch tensor.
                    If input_siganl is not 3d.
                    If size of dim2 (last) dim of input_signal is not 2.

        """
        if type(input_signal) != torch.Tensor:
            raise TypeError("input_signal must be an 3d torch tensor of floats. Got: %s"%(type(input_signal)))
        if len(input_signal.size()) != 3 or input_signal.dtype != torch.float:
            raise ValueError("input_signal must be an 3d torch tensor of floats. Got size: %s, dtype: %s"%(input_signal.size(), input_signal.dtype))
        if input_signal.size(2) != 2:
            raise ValueError("Dim 2 of input_signal, must have size 2. got dim2: %d"%(input_signal.size(2)))
        
        """
        Calculate energy of the signal x, and calculate noise standard
        deviation.

        \sigma^{2} = P/10**(SNR_db/10)
        \sigma = sqrt(P/10**(SNR_db/10))

        """
        energy_signal = input_signal.square().sum(dim=2).sqrt()
        sigma = energy_signal.div(10**(snr_db/10)).sqrt()
        sigma_per_dim = sigma.repeat_interleave(2, dim=1)
        sigma_per_dim = sigma_per_dim.view(input_signal.size(0), input_signal.size(1), 2)
        sigma_per_dim = sigma_per_dim.div(2)

        # Generate noise 
        w = torch.randn(input_signal.size()).mul(sigma_per_dim)
        
        output_signal = input_signal + w
        return output_signal