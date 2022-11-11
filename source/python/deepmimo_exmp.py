import DeepMIMO
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm

#from https://github.com/sergiossc/analog-beamforming-v2i
from utils import get_precoder_combiner, gen_dftcodebook, beamsweeping2, beamsweeping3

from plot_utils import plot_pattern_2D
from plot_utils import plot_array_factor
from plot_utils import plot_codebook_2D

def general_parameters_conf(scenario : str, scenario_folder : str, num_paths : int)->dict:
    # Load the default parameters
    parameters = DeepMIMO.default_params()

    # Set scenario name
    parameters['scenario'] = scenario

    # Set the main folder containing extracted scenarios
    parameters['dataset_folder'] = scenario_folder

    # To only include 25 strongest paths in the channel computation, set
    parameters['num_paths'] = num_paths


    return parameters

def userequipment_conf(parameters,active_ue:np.array,first_row:int, last_row:int, 
                        antenna_shape:np.array, rotation:np.array=np.array([0,0,0]),
                        spacing:float=0.25, pattern:str='isotropic'):

    parameters['active_users'] = active_ue

    # To activate the user rows 1-5, set
    parameters['user_row_first'] = first_row
    parameters['user_row_last'] = last_row

    # To adopt a 4 element ULA in y direction, set
    parameters['ue_antenna']['shape'] = antenna_shape

    # Rotate array 30 degrees around z-axis
    parameters['ue_antenna']['rotation'] = rotation
    parameters['ue_antenna']['spacing'] = spacing
    parameters['ue_antenna']['radiation_pattern'] = pattern

def basestation_conf(parameters,active_bs:np.array,antenna_shape:np.array,
                        spacing:float=0.25,rotation:np.array=np.array([0,0,0]),
                        pattern:str='isotropic',bs2bs:int=1):

    # To activate only the first basestation, set
    parameters['active_BS'] = active_bs

    parameters['bs_antenna']['shape'] = antenna_shape
    parameters['bs_antenna']['spacing'] = spacing
    parameters['bs_antenna']['rotation'] = rotation
    parameters['bs_antenna']['radiation_pattern'] = pattern
    parameters['enable_BS2BS'] = bs2bs


def signal_conf(parameters, num_channels : int, num_subcarriers : int, sc_limit : int, 
                        sc_sampling : int, bandwith : float, rx_filter : float):

    # Frequency (OFDM) or time domain channels
    parameters['OFDM_channels'] = num_channels
    parameters['OFDM']['subcarriers'] = num_subcarriers
    parameters['OFDM']['subcarriers_limit'] = sc_limit
    parameters['OFDM']['subcarriers_sampling'] = sc_sampling
    parameters['OFDM']['bandwidth'] = bandwidth
    parameters['OFDM']['RX_filter'] = rx_filter

def ue_channel_matrix(grid_size : int, dataset):
    pass


def codeword_snr(tx_power:float, noise_power:float,codeword, channel_matrix)->float:
    snr = (tx_power*np.linalg.norm(np.matmul(codeword, channel_matrix))**2/
            noise_power*np.linalg.norm(codeword)**2)

    return snr

def codeword_gain(codeword,channel_matrix)->float:
    gain = np.linalg.norm(np.matmul(codeword, channel_matrix))**2

    return gain

    
if __name__ == '__main__':
    scenario = 'O1_60'
    scenario_folder = r'../../../scenarios/deepmimo'
    num_paths = 25

    parameters = general_parameters_conf(scenario, scenario_folder, num_paths)

    active_users = np.array([180])
    first_row = 1
    last_row = 1
    ue_shape = np.array([1,2,1])
    userequipment_conf(parameters, active_users, first_row, last_row, ue_shape)

    active_bs = np.array([5])
    bs_shape = np.array([2,2,1])
    basestation_conf(parameters, active_bs, bs_shape)

    num_channels = 1
    num_subcarriers = 512
    subcarrier_limit = 1 #32
    subcarrier_sampling = 1
    bandwidth = 0.05
    rx_filter = 0
    signal_conf(parameters, num_channels, num_subcarriers, subcarrier_limit, subcarrier_sampling, bandwidth, rx_filter)

    dataset = DeepMIMO.generate_data(parameters)

    precoder = {}
    combiner = {}
    dftcodebook_tx = {}
    dftcodebook_rx = {}

    Lambda = 3e8/60e9

    plot_array_factor(bs_shape, 0.25)

    # ======================= DFT CODEBOOK =============================
    dftcodebook_tx = gen_dftcodebook(bs_shape[0]*bs_shape[1])
    print(dftcodebook_tx)

    plot_codebook_2D(dftcodebook_tx, bs_shape, 0.25)

    dftcodebook_rx = gen_dftcodebook(ue_shape[1])
    # print(dftcodebook_rx)

    for i, bs in enumerate(active_bs):
        for j, ue in enumerate(active_users):

            print("BS {i} and UE {j}".format(i=bs,j=ue))

            path = dataset[i]['user']['paths'][j]
            channel_matrix = dataset[i]['user']['channel'][j]

            channel = np.matrix(channel_matrix[:,:,0])

            #p_estmax, cw_id_max = beamsweeping2(channel, dftcodebook_tx)
            #print(p_estmax, cw_id_max)

            p_estmax, cw_id_max_tx, cw_id_max_rx = beamsweeping3(channel, dftcodebook_tx, dftcodebook_rx)
            print(p_estmax, cw_id_max_tx, cw_id_max_rx)

            plot_pattern_2D(dftcodebook_tx[:,cw_id_max_tx],  #codeword
                    bs_shape,           #antenna dimensions
                    0.25,                   #relative distance between elements
                    np.pi/2,#path['DoD_theta'][0],                
                    path['DoD_phi'][0])

            plot_pattern_2D(dftcodebook_rx[:,cw_id_max_rx],  #codeword
                    ue_shape,           #antenna dimensions
                    0.25,                   #relative distance between elements
                    np.pi/2, #path['DoA_theta'][0],                #Phi
                    path['DoA_phi'][0])

            # ================= SVD PRECODING AND COMBINING ====================
            # print("precoder and combiner")
            precoder, combiner = get_precoder_combiner(channel)
            p_s = (precoder.conj() * channel.T) * combiner.conj()
            p_s = norm(p_s) ** 2
            print("SVD",p_s)

            #======================= LLOYD CODEBOOK ============================


            #======================= GAIN COMPARISON ===========================
            '''
            svd_gain = codeword_gain()
            dft_gain = codeword_gain()
            lloyd_gain = codeword_gain()

            print('SVD Gain : {svd}\nDFT Gain : {dft}\nLloyd Gain : {gla}'.format(svd=svd_gain,
                    dft=dft_gain, gla=lloyd_gain))
            '''
