import numpy as np
import matplotlib.pyplot as plt

for type,style in [('array', 'solid'), ('list','dashed')]:
    N, time = map(lambda x: np.log10(x.reshape((3,6))), np.loadtxt('problem1_{0:s}.out'.format(type), unpack=True))

    settings = [('red','front'),('orange','middle'),('blue','end')]
    
    for i in range(3):
        color,label = settings[i]
        coeffs = np.polyfit(N[i,:], time[i,:], 1)
        plt.plot(N[i,:], time[i,:], color=color, ls=style, label=label if type == 'array' else None)
        plt.plot(N[i,:], coeffs[0]*N[i,:] + coeffs[1], color=color, ls=style)

plt.title('List vs Array Scaling')
plt.xlabel(r'$\log_{10}N$')
plt.ylabel(r'$\log_{10}t$ (ms)')
plt.legend(loc='lower right', frameon=False)
plt.text(3.2, -3.1, 'array (solid)', color='black', fontsize=12)
plt.text(3.2, -3.4, 'list     (dashed)', color='black', fontsize=12)
plt.savefig('problem1.pdf')
