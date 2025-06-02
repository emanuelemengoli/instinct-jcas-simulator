#5G Params:
import math
C = 3e8
FC = 3.5e9 #Hz
W = 20e6 #Hz Channel Bandwidth
N = -125 #dBm/Hz background noise
ALPHA = 2.7 #lin-scale #urban scenario prev. 3.5
PW_TX = 46 #dBm
#L0 = 21.8 #dBm/1m
G_TX = 23 #dBi ==> to compute the L0 at 1m
L0 = 20* math.log10((4*math.pi*FC)/C) - G_TX  #dBm/1m
#MU_FADING = 0 
#SIGMA_FADING_SQRD = BS_NOM_PW #1/(2*BS_NOM_PW)
#tune for further plots
#DEBUG = False
Wx = 1000
Hy = 750