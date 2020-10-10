import matplotlib.pyplot as plt
from pyComm.trainer import alterateTrainer

m = 16
n = 1
bsize = 1024
snr_db = 10.0
explore_variance = 0.02

trainer = alterateTrainer(m, n)
logs = trainer.train(batch_size=bsize, 
                        snr_db=snr_db, 
                        epochs=300,
                        explore_variance=explore_variance,
                        tx_optimizer='adam',
                        rx_optimizer='adam',
                        rx_optimizer_config_dict=None,
                        tx_optimizer_config_dict=None,
                        verbose=True,
                        sample_size=30**3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(logs['rx_loss'], color='red', label='RX loss')
ax1.plot(logs['tx_loss'], color='orange', label='TX loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Iteration')
ax1.legend()

'''
ax2.plot(error_rate_perc, color='red', label='RX loss')
ax2.set_ylabel('Error Rate')
ax2.set_xlabel('Iteration')
ax2.legend()
'''

#plt.show()
