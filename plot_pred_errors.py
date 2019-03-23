import numpy as np
import matplotlib.pyplot as plt


# Social-GAN results

Dataset = 'eth'  # Pred Len: 12, ADE: 0.71, FDE: 1.30
sgan_ADEs = np.array([0.14925498, 0.22303616, 0.30422062, 0.3876869 , 0.47234946,
       0.55686486, 0.64273345, 0.7300713 , 0.8200613 , 0.9131093 ,
       1.0084816 , 1.1066711 ], dtype=np.float32)
sgan_FDEs = np.array([0.14925498, 0.29681733, 0.46658948, 0.6380859 , 0.8109995 ,
       0.9794418 , 1.1579448 , 1.3414361 , 1.5399815 , 1.7505412 ,
       1.9622037 , 2.1867552 ], dtype=np.float32)


Dataset = 'eth'  # Pred Len: 8, ADE: 0.58, FDE: 1.14
sgan_ADEs = np.array([0.14810441, 0.22008175, 0.30093798, 0.38881227, 0.48404112,
       0.58674765, 0.6965651 , 0.81386673], dtype=np.float32)
sgan_FDEs = np.array([0.14810441, 0.2920591 , 0.46265045, 0.6524352 , 0.86495644,
       1.1002803 , 1.3554695 , 1.6349785 ], dtype=np.float32)


Dataset = 'hotel'  #  Pred Len: 12, ADE: 0.48, FDE: 1.02
sgan_ADEs = np.array([0.07749731, 0.11618809, 0.1599477 , 0.20959032, 0.26461777,
       0.32456204, 0.389115  , 0.4580106 , 0.53073597, 0.6067048 ,
       0.6857093 , 0.76755035], dtype=np.float32)
sgan_FDEs = np.array([0.07749731, 0.15487888, 0.24746689, 0.3585182 , 0.48472753,
       0.6242834 , 0.7764328 , 0.94027996, 1.1125387 , 1.2904239 ,
       1.4757544 , 1.6678016 ], dtype=np.float32)


Dataset = 'hotel'  # Pred Len: 8, ADE: 0.37, FDE: 0.72
sgan_ADEs = np.array([0.09576079, 0.14897919, 0.20753002, 0.27231717, 0.342311  ,
       0.41687617, 0.4958425 , 0.5788796 ], dtype=np.float32)
sgan_FDEs = np.array([0.09576079, 0.20219757, 0.32463166, 0.46667868, 0.6222862 ,
       0.7897021 , 0.9696404 , 1.1601397 ], dtype=np.float32)


Dataset = 'univ' # Pred Len: 12, ADE: 0.56, FDE: 1.18
sgan_ADEs = np.array([0.0545138 , 0.09331206, 0.13944164, 0.1920819 , 0.25044644,
       0.3138464 , 0.38167697, 0.45342898, 0.5286547 , 0.60697013,
       0.6880595 , 0.771661  ], dtype=np.float32)
sgan_FDEs = np.array([0.0545138 , 0.13211033, 0.23170075, 0.35000268, 0.48390457,
       0.6308461 , 0.7886604 , 0.9556932 , 1.1304603 , 1.311809  ,
       1.4989533 , 1.6912766 ], dtype=np.float32)


Dataset = 'univ'  # Pred Len: 8, ADE: 0.34, FDE: 0.70
sgan_ADEs = np.array([0.05585302, 0.09441645, 0.14098477, 0.19470836, 0.25458357,
       0.31968993, 0.38925216, 0.46260843], dtype=np.float32)
sgan_FDEs = np.array([0.05585302, 0.13297987, 0.23412141, 0.35587913, 0.4940845 ,
       0.64522165, 0.80662566, 0.9761022 ], dtype=np.float32)


Dataset = 'zara1'  # Pred Len: 12, ADE: 0.34, FDE: 0.68
sgan_ADEs = np.array([0.04983864, 0.08318949, 0.12239192, 0.16647977, 0.21479173,
       0.26689166, 0.32244238, 0.3811702 , 0.44285628, 0.5073228 ,
       0.5744148 , 0.6440084 ], dtype=np.float32)
sgan_FDEs = np.array([0.04983864, 0.11654034, 0.20079678, 0.29874334, 0.40803948,
       0.52739114, 0.65574676, 0.79226506, 0.9363449 , 1.0875213 ,
       1.2453347 , 1.4095378 ], dtype=np.float32)


Dataset = 'zara1'  #  Pred Len: 8, ADE: 0.21, FDE: 0.41
sgan_ADEs = np.array([0.04259074, 0.07450379, 0.1137694 , 0.15900362, 0.20905702,
       0.2631812 , 0.32085928, 0.38174048], dtype=np.float32)
sgan_FDEs = np.array([0.04259074, 0.10641684, 0.19230063, 0.29470623, 0.40927064,
       0.5338022 , 0.6669275 , 0.80790895], dtype=np.float32)


Dataset = 'zara2'  #  Pred Len: 12, ADE: 0.31, FDE: 0.65
sgan_ADEs = np.array([0.04416621, 0.07082314, 0.10320944, 0.14062566, 0.18227819,
       0.22755598, 0.2759604 , 0.32709056, 0.38062066, 0.43628556,
       0.49385193, 0.55313283], dtype=np.float32)
sgan_FDEs = np.array([0.04416621, 0.09748007, 0.16798201, 0.25287428, 0.34888837,
       0.45394492, 0.5663868 , 0.68500185, 0.80886143, 0.9372696 ,
       1.0695155 , 1.2052226 ], dtype=np.float32)


# Dataset= 'zara2'  # Pred Len: 8, ADE: 0.21, FDE: 0.43
# sgan_ADEs = np.array([0.04372119, 0.074482  , 0.1115614 , 0.15404812, 0.20113437,
#        0.25218958, 0.30673867, 0.36438257], dtype=np.float32)
# sgan_FDEs = np.array([0.04372119, 0.10524282, 0.18572018, 0.28150827, 0.3894794 ,
#        0.50746554, 0.6340333 , 0.76789004], dtype=np.float32)

# FIXME => Select Markers: https://matplotlib.org/api/markers_api.html

xLabels = [(i+1) * 0.4 for i in range(len(sgan_ADEs))]
plt.plot(xLabels, sgan_ADEs, '--v', label='S-GAN ADE')
plt.plot(xLabels, sgan_ADEs, '--s', label='S-GAN ADE')
plt.plot(xLabels, sgan_FDEs, '--^', label='S-GAN FDE')
plt.plot(xLabels, sgan_ADEs * 0.8, '--X', label='Ours ADE')
plt.plot(xLabels, sgan_FDEs * 0.8, '--2', label='Ours FDE')

plt.title(Dataset)
plt.grid(linestyle='-.', linewidth=1)
plt.xlim(0, xLabels[-1] + 0.2)
plt.xlabel('Predict Ahead Interval (s)')
plt.ylabel('Error (m)')
plt.legend()
plt.show()