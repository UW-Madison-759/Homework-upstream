import numpy as np
import matplotlib.pyplot as plt

def plot_timings(x, y, label, color, pos):
    coeffs = np.polyfit(x, y, 1)
    plt.plot(x, y, color=color, label=label, ls='solid')
    plt.plot(x, coeffs[0] * x + coeffs[1], color=color, ls='dashdot')
    plt.text(*pos, r'${0:.2f}\log_{{10}}N\,{1:.2f}$'.format(coeffs[0], coeffs[1]))

N, my_sort, std_sort = map(lambda x: np.log10(x), np.loadtxt('problem1c.out', unpack=True))
plot_timings(N, my_sort, 'my_sort', 'red', (4.5, 2.9))
plot_timings(N, std_sort, 'std_sort', 'blue', (4.5, -0.2))
plt.xlabel(r'$\log_{10} N$')
plt.ylabel(r'$\log_{10}time$ (ms)')
plt.title('Sort Comparison')
plt.legend(loc='upper left')
plt.savefig('problem1c.pdf')