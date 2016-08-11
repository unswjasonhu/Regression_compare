import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import colormaps as cmps
from matplotlib import cm
from matplotlib import style
style.use("ggplot")
import pandas as pd
import os

path = os.getcwd()

def scipy_idw(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)

def read_data(model):
    df = pd.read_csv(path+'/%s_populated_estimates.csv'%model, sep=',', header= None)
    print df.values.max()
    return df

def main():
    #Draw original
    df_original = pd.read_csv(path+'/original.csv',delimiter=',')
    row_original = df_original['grid_location_row1']
    col_original = df_original['grid_location_col1']
    val_original = df_original['co_original']
    xi = np.linspace(row_original.min(), row_original.max(), 100)
    yi = np.linspace(col_original.min(), col_original.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()
    grid = scipy_idw(row_original,col_original ,val_original, xi, yi)
    grid_original = grid.reshape((100, 100))

    vmax =20

    fig, axes = plt.subplots(nrows=2,ncols=4)
    #fig.subplots_adjust(right =0.8)
    ax1 = plt.subplot(2,4,1)
    im1 =plt.imshow(grid_original,cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(a)')

    ax2 = plt.subplot(2,4,2)
    im2 = plt.imshow(read_data('SVR'),cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(b)')

    ax3 = plt.subplot(2,4,3)
    im3 = plt.imshow(read_data('DTR'),cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(c)')

    ax4 = plt.subplot(2,4,4)
    im5 = plt.imshow(read_data('RFR'),cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(d)')

    ax5 = plt.subplot(2,4,5)
    im5 = plt.imshow(read_data('XGB'),cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(e)')

    ax6 = plt.subplot(2,4,6)
    im6 = plt.imshow(read_data('MLP'),cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(f)')

    ax7 = plt.subplot(2,4,7)
    im7 = plt.imshow(read_data('PLSR'),cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(g)')

    ax8 = plt.subplot(2,4,8)
    im8 = plt.imshow(read_data('LR'),cmap=cmps.plasma,vmin=0,vmax=vmax)
    plt.axis("off")
    plt.xlabel('(h)')

    cbar_ax=fig.add_axes([0.91,0.35,0.05,0.3])
    fig.colorbar(im1,cax=cbar_ax)
    #plt.savefig(path+'Fig/compare.png',format='png',dpi=200)
    plt.show()



    '''
    ax1.annotate('M5 Tunnel', xy=(70,80), xycoords='data',  xytext=(-30, -50), textcoords='offset points', fontsize=12,color='red',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='red',connectionstyle="arc3,rad=.2"))
    ax1.annotate('(a)', xy=(70,80), xycoords='data',  xytext=(-50, -50), textcoords='offset points', fontsize=12,color='black',
             arrowprops=dict(arrowstyle="->", linewidth = 0,color ='black',connectionstyle="arc3,rad=.2"))
    plt.title("Sensing Data",fontsize=12)

    ax2 = plt.subplot(1,3,3)
    im2 = plt.imshow(z11,cmap=cmps.plasma,vmin=0,vmax=48)
    plt.axis("off")
    ax2.annotate('M5 Tunnel', xy=(70,80), xycoords='data',  xytext=(-30, -50), textcoords='offset points', fontsize=12,color='red',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='red',connectionstyle="arc3,rad=.2"))
    ax2.annotate('CBD', xy=(70,30), xycoords='data',  xytext=(-70, 20), textcoords='offset points', fontsize=12,color='red',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='red',connectionstyle="arc3,rad=.2"))
    ax2.annotate('Eastern Distributor Motorway', xy=(90,50), xycoords='data',  xytext=(-150, 0), textcoords='offset points', fontsize=8,color='red',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='red',connectionstyle="arc3,rad=.2"))
    ax2.annotate('(c)', xy=(70,80), xycoords='data',  xytext=(-50, -50), textcoords='offset points', fontsize=12,color='black',
             arrowprops=dict(arrowstyle="->", linewidth = 0,color ='black',connectionstyle="arc3,rad=.2"))
    plt.title("SVR Estimation Data",fontsize=12)
    plt.xlabel('(b)')

    ax2 = plt.subplot(1,3,2)
    im2 = plt.imshow(z22,cmap=cmps.plasma,vmin=0,vmax=48)
    plt.axis("off")
    ax2.annotate('M5 Tunnel', xy=(70,80), xycoords='data',  xytext=(-30, -50), textcoords='offset points', fontsize=12,color='red',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='red',connectionstyle="arc3,rad=.2"))
    ax2.annotate('CBD', xy=(70,30), xycoords='data',  xytext=(-70, 20), textcoords='offset points', fontsize=12,color='red',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='red',connectionstyle="arc3,rad=.2"))
    ax2.annotate('Eastern Distributor Motorway', xy=(90,50), xycoords='data',  xytext=(-150, 0), textcoords='offset points', fontsize=8,color='red',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='red',connectionstyle="arc3,rad=.2"))
    ax2.annotate('(b)', xy=(70,80), xycoords='data',  xytext=(-50, -50), textcoords='offset points', fontsize=12,color='black',
             arrowprops=dict(arrowstyle="->", linewidth = 0,color ='black',connectionstyle="arc3,rad=.2"))
    plt.title("ANN Estimation Data",fontsize=12)
    plt.xlabel('(c)')

    cbar_ax=fig.add_axes([0.82,0.35,0.05,0.3])
    fig.colorbar(im1,cax=cbar_ax)
    #plt.savefig(path+'Fig/compare.png',format='png',dpi=200)
    #plt.show()
    '''
    '''
    err = 0
    for i in range(100):
        for j in range(100):
            err += abs(z11[i,j]- grid2[i,j]) ** 2

    print (err/10000) ** (0.5)
    '''
if __name__ == "__main__":
    main()