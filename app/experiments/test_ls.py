import numpy as np
from simulator.channelv2 import channel_version2
from models.ls_estimator import LS_estimator
from dataset.pilot_signal import pilot_gen


Nt=4
T=32
env=channel_version2(nt=Nt)
pilot=pilot_gen(Nt,T)
ls=LS_estimator()

X=pilot.signal()

h=env.channel_effect()

noise=0.01*(np.random.randn(T)+1j*np.random.randn(T))

Y=X@h + noise

h_hat=ls.estimate(X,Y)

print("True channel:", h)
print("Estimated channel:", h_hat)