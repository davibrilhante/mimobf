import DeepMIMO
import numpy as np
import matplotlib.pyplot as plt


def general_parameters_conf(scenario : str, scenario_folder : str, num_paths : int):
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
                        spacing:float=0.5, pattern:str='isotropic'):

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
                        spacing:float=0.5,rotation:np.array=np.array([0,0,0]),
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
    
if __name__ == '__main__':
    scenario = 'O1_60'
    scenario_folder = r'../scenarios/deepmimo'
    num_paths = 25

    parameters = general_parameters_conf(scenario, scenario_folder, num_paths)

    first_row = 1
    last_row = 1
    ue_shape = np.array([1,1,1])
    active_users = np.array([1,2,3])
    userequipment_conf(parameters, active_users, first_row, last_row, ue_shape)

    active_bs = np.array([1,2,3])
    bs_shape = np.array([1,4,1])
    basestation_conf(parameters, active_bs, bs_shape)

    num_channels = 1
    num_subcarriers = 512
    subcarrier_limit = 32
    subcarrier_sampling = 1
    bandwidth = 0.05
    rx_filter = 0
    signal_conf(parameters, num_channels, num_subcarriers, subcarrier_limit, subcarrier_sampling, bandwidth, rx_filter)

    dataset = DeepMIMO.generate_data(parameters)

    for i in active_bs:
        for j in active_users:
            print("BS {i} and UE {j}".format(i=i,j=j))
            channel_matrix = dataset[i-1]['user']['channel'][j]
            print(channel_matrix)
