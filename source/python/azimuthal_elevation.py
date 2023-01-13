import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

if __name__ == "__main__":
    d = 0.25
    ant = [1,10,1]
    k = 2*np.pi
    nsamples = 1000



    theta = np.arange(0, 2*np.pi, 2*np.pi/nsamples)
    phi = np.arange(0, 2*np.pi, 2*np.pi/nsamples)


    fig, axs = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(wspace=0.6, hspace=0.2, top=1.05, bottom=0.0)

    axs = axs.flat


    AZ = []



    #Azimuthal array factor with theta = np.pi/2 ==> np.sin(theta)=1
    elem_x = np.arange(ant[0])
    elem_y = np.arange(ant[1])
    elem_z = np.arange(ant[2])

    for p in phi:
        AF_x = np.exp(1j*elem_x*k*d*np.cos(p))
        AF_y = np.exp(1j*elem_y*k*d*np.sin(p))

        AZ.append(sum(np.kron(AF_x, AF_y)))

    #Azimuthal Array Factor
    AZ = 20*np.log10(AZ) - 20*np.log10(ant[0]*ant[1])

    axs[0].plot(phi, AZ)
    axs[0].set_rmin(-20)
    axs[0].set_rticks([-15,-10,-5,0]) #hide radial ticks
    axs[0].set_title('Azimuth XY')
    axs[0].grid(True)



    AF_z = []
    for t in theta:
        AF_z.append(sum(np.exp(1j*elem_z*k*d*np.cos(t))))

    #Elevation Array Factor
    EL = 20*np.log10(AF_z) - 20*np.log10(ant[2])

    axs[1].plot(theta, EL)
    axs[1].set_theta_zero_location("N")
    axs[1].set_rmin(-20)
    axs[1].set_rticks([-15,-10,-5,0]) #hide radial ticks
    axs[1].set_title('Elevation')
    axs[1].grid(True)

    AF_z = []
    for t in theta:
        AF_z.append(sum(np.exp(1j*elem_z*k*d*np.sin(t))))

    #Elevation Array Factor
    EL = 20*np.log10(AF_z) - 20*np.log10(ant[2])

    axs[2].plot(theta, EL)
    axs[2].set_theta_zero_location("N")
    axs[2].set_rmin(-20)
    axs[2].set_rticks([-15,-10,-5,0]) #hide radial ticks
    axs[2].set_title('Elevation XZ')
    axs[2].grid(True)


    PHI, THETA = np.meshgrid(phi, theta)
    AF_3D = np.zeros((nsamples,nsamples))

    for t in range(nsamples):
        for p in range(nsamples):
            AF_x = np.exp(1j*elem_x*k*d*np.sin(THETA[t][p])*np.cos(PHI[t][p]))
            AF_y = np.exp(1j*elem_y*k*d*np.sin(THETA[t][p])*np.sin(PHI[t][p]))

            AF_3D[t][p] = sum(np.kron(AF_x,AF_y))


    R = (1/(ant[0]*ant[1]*ant[2]))*np.array(AF_3D)
    #R = 20*np.log10(np.array(AF_3D)) - 20*np.log10(ant[0]*ant[1]*ant[2])
    #R = 20*np.log10(R)
    X = R*np.sin(THETA)*np.cos(PHI)
    Y = R*np.sin(THETA)*np.sin(PHI)
    Z = R*np.cos(THETA)

    #axs[3] = Axes3D(axs[3])
    axs[3] = plt.subplot(224,projection='3d')
    #axs[3] = fig.add_subplot(4,2,2,projection='3d')
    #axs[3].set_xlim3d(-1,1)
    #axs[3].set_ylim3d(-1,1)
    #axs[3].set_zlim3d(-1,1)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    my_col = plt.cm.jet(abs(np.real(Z)/np.amax(np.real(Z))))

    surf = axs[3].plot_surface(X, Y, Z, rstride=2, cstride=2, facecolors = my_col,
            linewidth=5, alpha=0.7)#antialiased=False)
    '''
    fig.text(0.5, 0.95, 'Azimuthal gain pattern of two element isotropic array',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    '''
    #plt.tight_layout()
    plt.show()
