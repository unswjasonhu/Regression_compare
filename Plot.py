import matplotlib.pyplot as plt
import pandas as pd
import os
plt.style.use('ggplot')

path = os.getcwd()
#Read estimates
df = pd.read_csv(path + '/estimate.csv', sep = ',')
#Set the figure size
fig = plt.figure()
fig.set_size_inches(28,14)
#Subplot the data
ax1 = fig.add_subplot(2, 4, 1)
fig1, = plt.plot(range(2844),df['SVR'], color= 'red')

ax2 = fig.add_subplot(2, 4, 2)
fig2, = plt.plot(range(2844),df['DTR'], color = 'blue')

ax3 = fig.add_subplot(2, 4, 3)
fig3, = plt.plot(range(2844),df['RFR'], color = 'purple')

ax4 = fig.add_subplot(2, 4, 4)
fig4, = plt.plot(range(2844),df['XGB'], color = 'green')

ax5 = fig.add_subplot(2, 4, 5)
fig5, = plt.plot(range(2844),df['MLP'], color = 'pink')

ax6 = fig.add_subplot(2, 4, 6)
fig6, = plt.plot(range(2844),df['PLSR'], color = 'orange')

ax7 = fig.add_subplot(2, 4, 7)
fig7, = plt.plot(range(2844),df['LR'], color = 'cyan')

ax8 = fig.add_subplot(2, 4, 8)
fig8, = plt.plot(range(2844),df['co'], color = 'yellow')
#Set one legend for subplots
fig.legend((fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8),('SVR','DTR','RFR','XGB','MLP','PLSR','LR','Original'),\
           loc='upper center',prop={'size':10},bbox_to_anchor=(0.5,0.95),fancybox=True,shadow=True,ncol=8)
#Save plot
fig.savefig(path + '/compare.png',format='png',dpi = 1000)
plt.show()