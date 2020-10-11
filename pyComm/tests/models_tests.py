import torch
torch.manual_seed(0)
import numpy as np
from pyComm.models import *

def test_models_base():
    """
    Verifies the basic structure of basic
    tx and rx.
    """
    M = 256
    N = 128
    mod = base_rx_tx(m=M, n=N)
    assert mod.constellation_length == M
    assert mod.num_channel_uses == N

def test_tx_model_forward_train_mode_false():
    """
    Verifies the tx model forward functionality
    when training is disabled.
    """
    M = 256
    N = 128
    mod = tx(m=M, n=N)
    messages = torch.randint(0, M, size=(1024, ))
    complex_signal = mod.forward(messages=messages, train_mode=False)
    assert list(complex_signal.size()) == [1024, 128, 2]
    # Verify unit normalized
    assert complex_signal.square().sum(dim=2).sum() == 1024*N
    # Very require grad is false
    assert complex_signal.requires_grad == False

def test_tx_model_forward_train_mode_true():
    """
    Verifies the tx model forward functionality
    when training is enabled.
    """
    M = 256
    N = 128
    expr_var = 0.02
    mod = tx(m=M, n=N)
    messages = torch.randint(0, M, size=(1024, ))
    complex_signal_mean, signal_dist = mod.forward(messages=messages, train_mode=True,
                                                   explore_variance=expr_var)
    
    assert list(complex_signal_mean.size()) == [1024, 128, 2]

    # Verify signal mean is unit normalized.
    assert complex_signal_mean.square().sum(dim=2).sum() == 1024*N
    
    # Very require grad is false
    assert complex_signal_mean.requires_grad == True
    
    # Very noise variance
    xp = signal_dist.sample()
    error = complex_signal_mean - xp
    error = error.view(-1).abs().sum().item()
    assert error <= np.sqrt(expr_var)*1024*N

    # Verify sampled signal is not unit energy
    assert xp.square().sum(dim=2).sum() != 1024*N

def test_rx_model_forward_train_mode_false():
    """
    Verifies the rx model forward functionality
    when training is disabled.
    """
    M = 128
    N = 64
    mod = rx(m=M, n=N)
    input_signal = torch.randn(1024, N, 2)

    logits, probs = mod.forward(input_signal=input_signal,
                                train_mode=False)
    
    # Very output size of tensors
    assert list(logits.size()) == [1024, M]
    assert list(probs.size()) == [1024, M]

    # Verify probs
    assert probs.sum(dim=1).sum().item() == 1024

    # Verify requires grad
    assert logits.requires_grad == False
    assert probs.requires_grad == False

def test_rx_model_forward_train_mode_true():
    """
    Verifies the rx model forward functionality
    when training is disabled.
    """
    M = 128
    N = 64
    mod = rx(m=M, n=N)
    input_signal = torch.randn(1024, N, 2)

    logits, probs = mod.forward(input_signal=input_signal,
                                train_mode=True)
    
    # Very output size of tensors
    assert list(logits.size()) == [1024, M]
    assert list(probs.size()) == [1024, M]

    # Verify probs
    assert probs.sum(dim=1).sum().item() == 1024

    # Verify requires grad
    assert logits.requires_grad == True
    assert probs.requires_grad == True

test_rx_model_forward_train_mode_true()