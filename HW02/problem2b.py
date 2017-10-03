import numpy as np
import matplotlib.pyplot as plt

N, time = map(lambda x: np.log10(x), np.loadtxt('problem2b.out', usecols=[0,1], unpack=True))
 
coeffs = np.polyfit(N, time, 1)
plt.plot(N, time, color='black', ls='solid')
plt.plot(N, coeffs[0] * N + coeffs[1], color='black', ls='dashdot')
plt.text(4.5, -1.1, r'${0:.2f}\log_{{10}}N\,{1:.2f}$'.format(coeffs[0], coeffs[1]), fontsize=16)
 
plt.xlabel(r'$\log_{10} N$')
plt.ylabel(r'$\log_{10}time$ (ms)')
plt.title('Exclusive Scan')
plt.savefig('problem2b.pdf')