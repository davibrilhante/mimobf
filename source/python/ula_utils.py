import numpy as np
from numpy.linalg import svd, norm
from itertools import product
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker as mticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import cmath

def polar(m):
   c = m.item(0)
   return (abs(c), cmath.phase(c))

def ula_array_factor(antenna_size, element_dist, phase, phi, theta):

    array_factor = []
    k = 2*np.pi#/Lambda

    array_factor = np.exp(1j*np.arange(antenna_size[1])*(k*element_dist*np.cos(phi) + phase))

    #print(phi,array_factor)
    
    return array_factor

def plot_array_factor(antenna_size, element_dist, phase, theta=np.pi/2, samples=1000, save=False, figname=None):

    phi = np.arange(-np.pi, np.pi, 1/samples)
    antenna_gain = np.sqrt(1/(antenna_size[0]*antenna_size[1]))
    pattern = [] 

    for p in phi:
        #Uniform Linear Arrary (ULA) Case:
        array_factor = ula_array_factor(antenna_size, element_dist, phase, p, theta)

        pattern.append(np.linalg.norm(antenna_gain*sum(array_factor))**2)

    fig, ax = plt.subplots(1,1,subplot_kw={'projection': 'polar'})
    ax.plot(phi, pattern)
    #ax.set_rticks([]) #hide radial ticks
    ax.grid(True)
    ax.set_title("Antenna Array Factor")

    if save and figname is not None:
        plt.savefig(figname)

    plt.show()

def gen_dftcodebook(num_of_cw, oversampling_factor=None):
    if oversampling_factor is not None:
        tx_array = np.arange(num_of_cw * int(oversampling_factor))
        mat = np.matrix(tx_array).T * tx_array
        cb = (1.0/np.sqrt(num_of_cw)) * np.exp(1j * 2 * np.pi * mat/(oversampling_factor * num_of_cw))
    elif oversampling_factor is None:
        tx_array = np.arange(num_of_cw)
        mat = np.matrix(tx_array).T * tx_array
        cb =  (1.0/np.sqrt(num_of_cw)) * np.exp(-1j * 2 * np.pi * mat/num_of_cw)
    else:
        raise(f'Please chose \'None\' or int value for oversampling_factor')

    return cb[:, 0:num_of_cw]

def beamsweeping(ch, cb_tx, cb_rx):
    '''
        This is correct! Checked!!
    '''

    p_est_max = -np.Inf
    cw_id_max_tx = ''
    cw_id_max_rx = ''

    beampairs = []
    
    num_of_cw_tx = cb_tx.shape[1]
    num_of_cw_rx = cb_rx.shape[1]

    for i in range(num_of_cw_tx):
        for j in range(num_of_cw_rx):

            # print("CB_TX {i} and CB_RX {j}".format(i=i,j=j))

            cw_tx = cb_tx[i]
            cw_rx = cb_rx[j]

            #print("cw_tx_rx",cw_tx.shape,ch.T.shape,cw_rx.shape)

            p_s = (cw_tx.conj() * ch.T) * cw_rx.conj().T
            p_s = norm(p_s) ** 2

            beampairs.append([i,j,p_s])
            beampairs.sort(reverse=True, key=lambda x: x[2])

            if p_s > p_est_max:
                p_est_max = p_s
                cw_id_max_tx = i
                cw_id_max_rx = j

    #print(beampairs[:20])
    
    return p_est_max, cw_id_max_tx, cw_id_max_rx

def codeword_pattern(codeword, antenna_size, element_dist, phi, theta, norm=False):
    antenna_gain = np.sqrt(1/(antenna_size[0]*antenna_size[1]))
    phase = 0

    #Uniform Linear Array (ULA)
    array_factor = np.matrix(ula_array_factor(antenna_size, element_dist, phase, phi, theta))

    pattern = array_factor * np.matrix(codeword)

    if norm:
        pattern = np.linalg.norm(antenna_gain*pattern)**2
    else:
        pattern = np.linalg.norm(pattern)**2

    return pattern

def plot_pattern_2D(codeword, antenna_size, element_dist, theta=np.pi/2, marker=None, samples=1000, save=False, figname=None):
    phi = np.arange(-np.pi, np.pi, 1/samples)
    pattern = [] #[[] for i in range(antenna_size[0]*antenna_size[1])]

    for p in phi:
        pattern.append(codeword_pattern(codeword, antenna_size, element_dist, p, theta, True))


    fig, ax = plt.subplots(1,1,subplot_kw={'projection': 'polar'})

    ax.plot(phi, pattern)
    if marker is not None:
        ax.vlines(marker, 0, max(pattern), color='red')

    # ax.set_rticks([]) #hide radial ticks
    ax.grid(True)

    if save and figname is not None:
        plt.savefig(figname)

    plt.show()
    

def plot_codebook_2D(codebook, antenna_size, element_dist, theta=np.pi/2, samples=1000, save=False, figname=None):
    phi = np.arange(-np.pi, np.pi, 1/samples)

    size = antenna_size[0]*antenna_size[1]*antenna_size[2]
    fig, ax = plt.subplots(1,1,subplot_kw={'projection': 'polar'})

    for index in range(size): 
        codeword = codebook[:,index]
        #print(codeword)
        pattern = []

        for p in phi:
                pattern.append(codeword_pattern(codeword, antenna_size, element_dist, p, theta, True))

        ax.plot(phi, pattern, label = index)

    #ax.set_rticks([]) #hide radial ticks
    ax.set_title('Codebook Plot')
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    if save and figname is not None:
        plt.savefig(figname)

    plt.show()

