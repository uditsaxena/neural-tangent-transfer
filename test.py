import sys
sys.path.append("./../")
import matplotlib.pyplot as plt
from jax import random
import numpy as np
from nt_transfer import *
import numpy as onp
from nt_transfer.nn_models import model_dict
from nt_transfer.plot_tools import *
import matplotlib.ticker as ticker

if __name__ == '__main__':
    gen_kwargs = dict(dataset_str='mnist',
                      model_str='mlp_lenet',
                      NN_DENSITY_LEVEL_LIST=[0.03],  # the fraction of weight remainining
                      OPTIMIZER_STR='adam',  # the optimizer
                      NUM_RUNS=2,  # two independent runs (note that in our paper, we use NUM_RUNS = 5)
                      NUM_EPOCHS=20,  # number of epochs
                      BATCH_SIZE=64,  # batch size
                      STEP_SIZE=5e-4,  # SGD step size
                      MASK_UPDATE_FREQ=100,  # mask update frequency
                      LAMBDA_KER_DIST=1e-3,  # the strength constant for NTK distance used in NTT loss function
                      LAMBDA_L2_REG=1e-4,  # the l2 regularization constant
                      SAVE_BOOL=True,
                      save_dir='./../ntt_results/')

    print(os.getcwd())
    model = nt_transfer_model(**gen_kwargs, instance_input_shape=[784])
    _, _, nt_trans_vali_all_sparsities_all_runs = model.optimize()