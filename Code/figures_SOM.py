
import os
path='C:/Users/arne/Documents/GitHub/Masterproef2018Code/Plots'; #Change this to directory that stores the data
os.chdir(path)
import matplotlib.pyplot as plt

xdata=[1,2,3,4,0,5]
ydata=[2,1,4,3,0,5]
labels=['A', 'B', 'C', 'D', 'W1', 'W2']

f, ax1 = plt.subplots()

ax1.plot(xdata[0:4], ydata[0:4], 'ro')
ax1.axis([-1, 7, -1, 7])
ax1.plot(xdata[4:], ydata[4:], 'b^')
for label, x, y in zip(labels, xdata, ydata):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()
f.savefig('SOM_init.eps', format='eps', dpi=1000)

xdata=[1,2,3,4,1.5,3.5]
ydata=[2,1,4,3,1.5,3.5]
labels=['A', 'B', 'C', 'D', 'W1', 'W2']

f, ax1 = plt.subplots()

ax1.plot(xdata[0:4], ydata[0:4], 'ro')
ax1.axis([-1, 7, -1, 7])
ax1.plot(xdata[4:], ydata[4:], 'b^')
for label, x, y in zip(labels, xdata, ydata):
    plt.annotate(
        label,
        xy=(x, y), xytext=(20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()
f.savefig('SOM_end.eps', format='eps', dpi=1000)


