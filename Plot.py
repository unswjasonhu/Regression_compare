import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')

#Read estimates
df = pd.read_csv( 'estimate.csv', sep = ',')

#Set the figure size
fig = plt.figure()

fig.set_size_inches(28,14)
plt.yticks(())
plt.xticks(())
plt.xlabel('Test samples',fontsize = 36,labelpad=55)
plt.ylabel('CO concentrations (ppm)',fontsize = 36,labelpad=40)
#Subplot the data
ax1 = fig.add_subplot(2, 4, 1)
fig1, = plt.plot(range(285),df['SVR'], color= 'red')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(a)',fontsize = 28,labelpad=0)

ax2 = fig.add_subplot(2, 4, 2)
fig2, = plt.plot(range(285),df['DTR'], color = 'blue')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(b)',fontsize = 28,labelpad=0)

ax3 = fig.add_subplot(2, 4, 3)
fig3, = plt.plot(range(285),df['RFR'], color = 'purple')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(c)',fontsize = 28,labelpad=0)

ax4 = fig.add_subplot(2, 4, 4)
fig4, = plt.plot(range(285),df['XGB'], color = 'green')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(d)',fontsize = 28,labelpad=0)

ax5 = fig.add_subplot(2, 4, 5)
fig5, = plt.plot(range(285),df['MLP'], color = 'pink')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(e)',fontsize = 28,labelpad=0)

ax6 = fig.add_subplot(2, 4, 6)
fig6, = plt.plot(range(285),df['LR'], color = 'cyan')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(f)',fontsize = 28,labelpad=0)

ax7 = fig.add_subplot(2, 4, 7)
fig7, = plt.plot(range(285),df['ABR'], color = 'orange')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(g)',fontsize = 28,labelpad=0)

ax8 = fig.add_subplot(2, 4, 8)
fig8, = plt.plot(range(285),df['co_original'], color = 'yellow')
plt.xlim(-4,290)
plt.ylim(0,40)
plt.xticks([0,58,116,174,232,290],fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('(h)',fontsize = 28,labelpad=0)

#Set one legend for subplots
fig.legend((fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8),('SVR','DTR','RFR','XGB','MLP','LR','ABR','Original'),\
           loc='upper center',prop={'size':28},bbox_to_anchor=(0.52,0.96),fancybox=True,shadow=True,ncol=8)
#Save plot
fig.savefig('test_accuracy.png',format='png',dpi = 200)
#plt.show()