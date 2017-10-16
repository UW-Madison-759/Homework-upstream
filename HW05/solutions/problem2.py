import numpy as np
import matplotlib.pyplot as plt
import sys

name_trans = {
    'euler'     : 'Euler: Intel Broadwell; 2016',
    'skylake'   : 'Intel 2 core Skylake; 2014',
    'bulldozer' : 'AMD 8 core Bulldozer; 2011'
}

def plot(source,color):
    N, time = np.loadtxt('{0:s}_problem2.dat'.format(source), usecols=[0, 1], unpack=True)
    plt.loglog(N, time, color=color, ls='solid', label=name_trans[source], basex=2)
    N, time = np.loadtxt('{0:s}_problem2_unrolled.dat'.format(source), usecols=[0, 1], unpack=True)
    plt.loglog(N, time, color=color, ls='dashed', basex=2)
#---------------------------------------------------------------------------------   

plot('euler', 'blue')
plot('skylake', 'red')
plot('bulldozer', 'orange')
plt.xlim(0, 40)
plt.ylim(1.0, 40)
plt.legend(frameon=False)

plt.xlabel('# threads')
plt.ylabel('time (ms)')
plt.savefig('problem2.pdf')
