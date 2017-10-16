import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print('Usage: {0:s} input'.format(sys.argv[0]))
    exit()

N, time = np.loadtxt(sys.argv[1], usecols=[7,8], unpack=True)
plt.plot(N, time)
plt.title('euler01')
plt.xlabel('# threads')
plt.ylabel('time (ms)')
plt.savefig('problem1.pdf')
