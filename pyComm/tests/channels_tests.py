import torch
torch.manual_seed(0)
import numpy as np
from pyComm.channels import *

def test_awgn_channel_propogate():
    """
    Verifies the awgn channel propogate
    function.
    """
    awgn = AwgnChannel()
    for i in range(0, 10):
        input_signal = torch.randn(4, 2, 2)
        y = awgn.propagate(input_signal, snr_db=10.0)
        out = (y == input_signal).int().sum()
        assert out.item() == 0

def test_awgn_channel_at_different_snr():
    """
    Verifies the awgn channel noise at different 
    SNRs.
    """
    snrs_db = np.linspace(-30, 40, num=30)
    avg_error = []

    input_signal = torch.randn(5000, 100, 2)
    awgn = AwgnChannel()
    
    for snr in snrs_db:
        y = awgn.propagate(input_signal, snr_db=snr)
        error = y.sub(input_signal)
        error = error.view(-1).abs()
        avg_error.append(error.mean().item())
    assert sorted(avg_error, reverse=True) == avg_error

def get_noise_std_dev(snr_db, signal_power):
    """
    Returns the noise standard deviation
    given snr for the aditive noise.
    
    Parameters
    ----------
    * snr_db            : float
                            Signal-to-Noise ratio in dB.
    * signal_power      : float
                            Signal power in watts.
    """
    var = signal_power/ (10**(snr_db/10))
    return np.sqrt(var)
    

def test_awgn_channel_normalized_input_signal():
    """
    Verifies the propoate api when the input signal
    given is normalized to have unit energy per symbol.
    """
    awgn = AwgnChannel()
    input_signal = torch.randn(5000, 100, 2)
    snr = 10.0

    input_signal_normalized = torch.nn.functional.normalize(input_signal, p=2, dim=2)
    # Verify symbol is normalized
    assert input_signal_normalized.square().sum(dim=2).sum().item() == 500000

    snrs_db = np.linspace(-30, 40, num=30)
    for snr in snrs_db:
        y = awgn.propagate(input_signal_normalized, snr_db=snr)
        sigma = get_noise_std_dev(snr, 1) * 500000
        error = y - input_signal_normalized
        error = error.view(-1).abs()
        #print(error.sum().item(), sigma)
        assert error.sum().item() <= sigma