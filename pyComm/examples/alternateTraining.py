from pyComm.trainer import alterateTrainer

m = 16
n = 2
bsize = 1024
snr_db = 10.0

trainer = alterateTrainer(m, n)
trainer.train(batch_size=bsize, snr_db=snr_db)
