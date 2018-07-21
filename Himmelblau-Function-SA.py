import numpy as np
import matplotlib.pyplot as plt



x0 = 2 # Initial solution you'd like to start at
y0 = 1

k = 0.1
T0 = 1000
M = 300
N = 15
alpha = 0.85

#Objective function
z_int = ((x0**2)+y0-11)**2+(x0+(y0**2)-7)**2

print("Initial X is %.3f" % x0)
print("Initial Y is %.3f" % y0)
print("Initial Z is %.3f" % z_int)

#listas para hacer trazas de la optimización
temp = []
min_z = []

for i in range(M):
    for j in range(N):
        xt = 0
        yt = 0
        
        ran_x_1 = np.random.rand()
        ran_x_2 = np.random.rand()
        ran_y_1 = np.random.rand()
        ran_y_2 = np.random.rand()
        
        if ran_x_1 >= 0.5:
            x1 = k*ran_x_2
        else:
            x1 = -k*ran_x_2
        
        if ran_y_1 >= 0.5:
            y1 = k*ran_y_2
        else:
            y1 = -k*ran_y_2
            
        xt = x0+x1
        yt = y0+y1
        
        of_new = ((xt**2)+yt-11)**2+(xt+(yt**2)-7)**2
        
        of_current = ((x0**2)+y0-11)**2+(x0+(y0**2)-7)**2
        
        
        ran_1 = np.random.rand()
        form = 1/(np.exp((of_new-of_current)/T0))
        
        if of_new <= of_current:
            x0 = xt
            y0 = yt
        elif ran_1<=form:
            x0 = xt
            y0 = yt
        else:
            x0 = x0
            y0 = y0
        
    temp = np.append(temp,T0)
    min_z = np.append(min_z,of_current)
    T0 = alpha*T0


print("X is %.3f" % x0)
print("Y is %.3f" % y0)
print("Final OF is %.3f" % of_current)


plt.plot(temp,min_z)
plt.title("Z vs. Temp.",fontsize=20, fontweight='bold')
plt.xlabel("Temp.",fontsize=18, fontweight='bold')
plt.ylabel("Z",fontsize=18, fontweight='bold')

plt.xlim(1000,0)
plt.xticks(np.arange(min(temp),max(temp),100),fontweight='bold')
plt.yticks(fontweight='bold')

plt.show()









