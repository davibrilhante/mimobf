import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker as mticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def dft_codebook(antenna_size, bits=3):
    seq = np.matrix(np.arange(antenna_size[0]*antenna_size[1]))
    mat = seq.conj().T * seq
    codebook = np.exp(-2j*np.pi*mat/(antenna_size[0]*antenna_size[1]))
    return codebook

def gen_codebook(num_of_cw):
    tx_array = np.arange(num_of_cw)
    mat = np.matrix(tx_array).T * tx_array
    cb = np.exp(1j * 2 * np.pi * mat/num_of_cw)
    return cb

def upa_array_factor(antenna_size, Lambda, element_dist, phi, theta):
    norm = np.sqrt(1/(antenna_size[0]*antenna_size[1]))
    array_factor = []
    k = 2*np.pi/Lambda


    x_factor = np.exp(1j*k*element_dist*np.arange(antenna_size[0])*np.sin(theta)*np.cos(phi))
    y_factor = np.exp(1j*k*element_dist*np.arange(antenna_size[1])*np.cos(theta))

    return np.linalg.norm(norm*sum(np.kron(x_factor, y_factor)))

def upa_pattern(antenna_size, Lambda, element_dist, phi, theta):

    antenna_gain = np.sqrt(1/(antenna_size[0]*antenna_size[1]))
    k = 2*np.pi/Lambda

    codebook = dft_codebook(antenna_size)

    x_factor = np.exp(1j*k*element_dist*np.arange(antenna_size[0])*np.sin(theta)*np.cos(phi))
    y_factor = np.exp(1j*k*element_dist*np.arange(antenna_size[1])*np.sin(theta)*np.sin(phi))

    array_factor = np.matrix(np.kron(y_factor,x_factor))

    channel = array_factor.T

    pattern = np.linalg.norm(codebook.conj().T*channel, axis=1)


    return pattern


def get_polar_pattern_2D(antenna_size, Lambda, element_dist, phi=np.pi/6, samples=100):
    theta = np.arange(-np.pi, np.pi, 1/samples)
    pattern = [[] for i in range(antenna_size[0]*antenna_size[1])]

    for t in theta:
        temp = upa_pattern(antenna_size, Lambda, element_dist, phi, t)
        for n, p in enumerate(temp):
            pattern[n].append(p)


    return theta, pattern

def codeword_pattern(codeword, antenna_size, Lambda, element_dist, phi, theta):
    antenna_gain = np.sqrt(1/(antenna_size[0]*antenna_size[1]))
    k = 2*np.pi/Lambda
    x_factor = np.exp(1j*k*element_dist*np.arange(antenna_size[0])*np.sin(theta)*np.cos(phi))
    y_factor = np.exp(1j*k*element_dist*np.arange(antenna_size[1])*np.sin(theta)*np.sin(phi))

    array_factor = np.matrix(np.kron(y_factor,x_factor))

    channel = array_factor.T
    #print(codeword.shape, channel.shape)
    pattern = np.linalg.norm(codeword.conj().T * channel)**2

    return pattern

def plot_codeword_2D(codeword, antenna_size, Lambda, element_dist, theta=np.pi/2, marker=None, samples=100, save=False, figname=None):
    phi = np.arange(-np.pi, np.pi, 1/samples)
    pattern = [] #[[] for i in range(antenna_size[0]*antenna_size[1])]

    for p in phi:
        pattern.append(codeword_pattern(codeword, antenna_size, Lambda, element_dist, p, theta))


    fig, ax = plt.subplots(1,1,subplot_kw={'projection': 'polar'})

    ax.plot(phi, pattern)
    if marker is not None:
        ax.vlines(marker, 0, max(pattern), color='red')

    ax.set_rticks([]) #hide radial ticks
    ax.grid(True)

    if save and figname is not None:
        plt.savefig(figname)

    plt.show()
    

def get_polar_pattern_3D(antenna_size, Lambda, element_dist, samples=200):
    codebook = dft_codebook(antenna_size)

    theta = np.linspace(-np.pi, np.pi, samples)
    phi= np.linspace(0, 2*np.pi, samples)

    PHI, THETA = np.meshgrid(phi, theta)
    AF = [[0 for j in range(samples)] for i in range(samples)]

    k = 2*np.pi/Lambda

    for t in range(samples):
        for p in range(samples):
            Sx = np.exp(1j*k*element_dist*np.arange(antenna_size[0])*np.sin(THETA[t][p])*np.cos(PHI[t][p]))
            Sy = np.exp(1j*k*element_dist*np.arange(antenna_size[1])*np.sin(THETA[t][p])*np.sin(PHI[t][p]))
            #AF[t][p] = sum(np.kron(Sx, Sy))
            product = np.matrix(np.kron(Sx, Sy))
            channel = product.T #product.conj().T*product
            AF[t][p] = np.linalg.norm(codebook.conj().T[0]*channel)

    R = (1/antenna_size[0])*(1/antenna_size[1])*np.array(AF)

    X = R*np.sin(THETA)*np.cos(PHI)
    Y = R*np.sin(THETA)*np.sin(PHI)
    Z = R*np.cos(THETA)

    return X, Y, Z


    

if __name__ == "__main__":
    antenna_size = [1,64]
    freq = 60e9
    Lambda = 3e8/freq
    element_dist = 0.25*Lambda

    codebook = dft_codebook(antenna_size)

    theta, r = get_polar_pattern_2D(antenna_size, Lambda, element_dist)

    aspect_v = int(np.sqrt(antenna_size[0]*antenna_size[1]))
    if antenna_size[0]*antenna_size[1]%aspect_v == 0:
        aspect_h = aspect_v
    elif antenna_size[0]*antenna_size[1]%aspect_v > 0:
        aspect_h = aspect_v + 1

    fig, ax = plt.subplots(aspect_v,aspect_h,
                            subplot_kw={'projection': 'polar'})

    print(np.shape(ax))
    '''

    fig.subplots_adjust(left=0.06,
                    bottom=0.41,
                    right=0.95,
                    top=1.0,
                    wspace=0,
                    hspace=0)
    '''

    for v in range(aspect_v):
        for h in range(aspect_h):
            try:
                pattern = r[v*aspect_v+h]
            except IndexError:
                break

            peak = max(pattern)

            ax[v,h].plot(theta, pattern)

            ax[v,h].set_rticks([])#[peak/4, peak/2, 3*peak/4, peak])  # Less radial ticks
            #ax[int(n/4),n%4].set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax[v,h].grid(True)

            ticks_loc = ax[v,h].get_xticks().tolist()
            ax[v,h].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax[v,h].set_xticklabels(['0°']+['' for i in range(7)])

    plt.tight_layout()
    plt.savefig('multiplot.eps')
    plt.show()

    fig = plt.figure()
    fig.subplots_adjust(wspace=0)
    ax = Axes3D(fig)
    Y, Z, X = get_polar_pattern_3D(antenna_size, Lambda, element_dist, 200)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    #ax.set_axis_off()
    plt.xlabel('X')
    plt.ylabel('Y')
    my_col = plt.cm.jet(abs(np.real(Z)/np.amax(np.real(Z))))

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col,
            linewidth=5, alpha=0.7)#antialiased=False)

    plt.show()
