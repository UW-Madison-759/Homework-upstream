import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print('Usage: {0:s} input output.pdf'.format(sys.argv[0]))
    exit()

plt.title('euler01')
plt.xlabel('# threads')
plt.ylabel('time (ms)')
plt.savefig(sys.argv[2])
