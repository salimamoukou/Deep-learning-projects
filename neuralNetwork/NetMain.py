from Network import *
from NetworkArchi import *
import numpy as np
# ""test""



feat = data_load['features']
label = data_load['label']


print(feat.shape)
#
# inp = np.reshape(np.random.uniform(size=65*5*3),(65,5,3))
# out = np.reshape(np.random.uniform(size=65*5*1),(65,5,1))
#
archi = NetworkArchi()
network = Network(archi)
network.train(feat, label)
