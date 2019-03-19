import numpy as np
import matplotlib.pyplot as plt


# Social-GAN results
dataset_name = 'ETH'
sgan_ADEs = np.array([0.14685781, 0.2207984, 0.30235082, 0.38624543, 0.47160196, 0.55735147,
                      0.6449313, 0.73441803, 0.8263909, 0.92123747, 1.0185804, 1.1188059])
sgan_FDEs = np.array([0.14685781, 0.29473898, 0.46545565, 0.6379292, 0.8130281, 0.9860988,
                      1.1704108, 1.3608246, 1.5621741, 1.7748566, 1.9920095, 2.2212863])

xLabels = [(i+1) * 0.4 for i in range(len(sgan_ADEs))]
plt.plot(xLabels, sgan_ADEs, '--^', label='S-GAN ADE')
plt.plot(xLabels, sgan_FDEs, '--+', label='S-GAN FDE')
plt.plot(xLabels, sgan_ADEs * 0.8, '--*', label='Ours ADE')
plt.plot(xLabels, sgan_FDEs * 0.8, '--o', label='Ours FDE')

plt.grid(linestyle='-.', linewidth=1)
plt.xlim(0, xLabels[-1] + 0.2)
plt.xlabel('Predict Ahead Interval (s)')
plt.ylabel('Error (m)')
plt.legend()
plt.show()