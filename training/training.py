import itertools

import pandas as pd
from pandas import DataFrame

import numpy as np

import os


#######################
# BUILDING THE HEADER #
#######################
LABELS = [
    # Kistler force plates
    'Fx_l','Fy_l','Fz_l', 'M_l',
    'Fx_r','Fy_r','Fz_r', 'M_r'
]
INPUTS = [
    # Moticon insoles
    'Ftot_l', 'CoPx_l', 'CoPy_l',
    'Ftot_r', 'CoPx_r', 'CoPy_r',

    'P1_l', 'P2_l', 'P3_l', 'P4_l', 'P5_l', 'P6_l', 'P7_l', 'P8_l', 'P9_l', 'P10_l', 'P11_l', 'P12_l', 'P13_l', 'P14_l', 'P15_l', 'P16_l',
    'ax_l', 'ay_l', 'az_l',
    'angx_l', 'angy_l', 'angz_l',

    'P1_r', 'P2_r', 'P3_r', 'P4_r', 'P5_r', 'P6_r', 'P7_r', 'P8_r', 'P9_r', 'P10_r', 'P11_r', 'P12_r', 'P13_r', 'P14_r', 'P15_r', 'P16_r',
    'ax_r', 'ay_r', 'az_r',
    'angx_r', 'angy_r', 'angz_r',

    # Cometa EMG sensors
    'vm_r',  'vm_l',  # vastus medialis
    'vr_r',  'vr_l',  # vastus radialis
    'gm_r',  'gm_l',  # gluteus medius
    'tfl_r', 'tfl_l' # tensor fasciae latae
]
MASKS = [
    'fp1', 'fp2', 'fp3', 'fp4',
    'valid_mask_feet', # 0 = occluded markers
    'correct_mask_fp', # 0 = force plate gave bad prediction
    'correct_mask_ins' # 0 = foot incorrectly placed on force plate
]
HEADER = LABELS + INPUTS + MASKS


####################
# LOADING THE DATA #
####################
DATA_DIR = "../segmented_data/"
SUBJECTS = ['RB']
SCENES = ['FlatWalkStraight']
TRIALS = ('FW walking')

# Reads the csv-files in ./segmented_data of the SUBJECTS-SCENES-TRIALS combinations specified above
def read_gait_cycles():
    # Adds the features of several past and future timesteps to each row
    def expand_timeframe(gait_cycle):
        NR_OF_TIMESTEPS = 5

        past   = [ 2**i for i in range(NR_OF_TIMESTEPS)]
        future = [-2**i for i in range(NR_OF_TIMESTEPS)]

        for offset in past + future:
            # Shift the relevant columns back/forwards in time
            columns = [col for col in INPUTS if not col.startswith('P')]
            shifted = gait_cycle[columns].shift(offset)

            # Rename shifted columns and concat to gait_cycle
            timestep = offset * -2 # ms
            shifted.columns = [f'{col}_{timestep}ms' for col in columns]
            gait_cycle = pd.concat([gait_cycle, shifted], axis=1)

        return gait_cycle

    # Removes the first and final (25) rows for which no target label is available
    def remove_buffers(df):
        cond = df.iloc[:, 0:8].sum(axis=1) != 0
        first = df[cond].index[0]
        last = df[cond].index[-1] + 1
        return df[first:last]

    # Reads a single csv-file, expands the features and removes the buffers
    def read_gait_cycle(filepath):
        gait_cycle = pd.read_csv(filepath, header=None, names=HEADER)
        gait_cycle = expand_timeframe(gait_cycle)
        return remove_buffers(gait_cycle)


    # Build up DataFrame of gait cyles by traversing the specified TRIALS
    df = DataFrame()

    for subdirs in itertools.product(SUBJECTS, SCENES):
        path = DATA_DIR + '/'.join(subdirs)

        for file in os.listdir(path):
            if file.startswith(TRIALS) and file.endswith('.csv'):
                filepath = path + '/' + file
                gait_cycle = read_gait_cycle(filepath)
                df = pd.concat([df, gait_cycle])

    return df

df = read_gait_cycles()
