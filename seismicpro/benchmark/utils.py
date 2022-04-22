""" SeismicPro-specific functions for Benchmark """

import os
import sys
sys.path.append('../..')

import numpy as np
from seismicpro import Survey, make_prestack_segy

def make_benchmark_data(path):
    """ Generate a SEGY file with specific geometry for benchmark and save it to disk. """
    # The geometry defined below should be changed only together with survey filtering parameters
    # to ensure that after filtering all the gathers \ supergathers have the same number of traces
    if not os.path.isfile(path):
        make_prestack_segy(path, survey_size=(400, 400), sources_step=(5, 5), receivers_step=(5, 5),
                           activation_dist=(50, 50), bin_size=(10, 10))

def load_benchmark_data(path):
    """ Load data from file and filter it to obtain survey objects for benchmark. """
    # Load headers and add synthetic FirstBreak times
    sur = Survey(path, header_index=['INLINE_3D', 'CROSSLINE_3D'],
                 header_cols='offset', name='raw')
    sur.headers['FirstBreak'] = np.random.randint(0, 3000, len(sur.headers))

    # Drop three lines of CDPs from each side of the survey, since they have less traces than central ones
    cl_min, cl_max = sur['CROSSLINE_3D'].min(), sur['CROSSLINE_3D'].max()
    il_min, il_max = sur['INLINE_3D'].min(), sur['INLINE_3D'].max()
    survey = (sur.filter(lambda x: (x>cl_min+3) & (x<=cl_max-3), 'CROSSLINE_3D')
                 .filter(lambda x: (x>il_min+3) & (x<=il_max-3), 'INLINE_3D'))

    sg_survey = survey.generate_supergathers((3,3), (1,1), (0,0))
    # Drop one line of supergathers from each side of the survey, since they have less traces than central ones
    sgc_min, sgc_max = sg_survey['SUPERGATHER_CROSSLINE_3D'].min(), sg_survey['SUPERGATHER_CROSSLINE_3D'].max()
    sgi_min, sgi_max = sg_survey['SUPERGATHER_INLINE_3D'].min(), sg_survey['SUPERGATHER_INLINE_3D'].max()
    sg_survey = (sg_survey.filter(lambda x: (x>sgc_min+1) & (x<=sgc_max-1), 'SUPERGATHER_CROSSLINE_3D')
                           .filter(lambda x: (x>sgi_min+1) & (x<=sgi_max-1), 'SUPERGATHER_INLINE_3D'))

    return survey, sg_survey
