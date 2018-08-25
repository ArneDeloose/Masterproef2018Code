#Plot soundwave
import AD_functions as AD
import matplotlib.pyplot as plt
AD.path();
sample_rate, samples, t, total_time,steps= AD.spect('Test.wav');
plt.plot(t,samples)
plt.title('Soundwave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout() #Ensures axis doesn't get cut off when saving
plt.show()
#plt.savefig('Test')