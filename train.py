import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Suppress Tensorflow Messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CPU/GPU
import numpy as np
from agent import A2C
from env import Environment
from model import Model


'''Log directory'''
verbose = True
if verbose:
    save_dir = os.path.join(os.getcwd()+'/Results', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_dir)
    log_file_name = save_dir+'/log.txt'
    reward_file_name = save_dir+'/rewards'
    loss_file_name = save_dir+'/loss'
else:
    log_file_name = ""

'''Environment Parameters'''
game = 'MiniGrid-DoorKey-6x6-v0'
seed = 0  # Seed for Env, TF, Numpy
num_frames = 5e6  # Million Frames
logs = {'log_interval': 50,  # Number of Episodes after which to print output/save batch output
        'log_file_name': log_file_name
        }

'''Parameters of Algorithm'''
algorithm_parameters = {'batch_size': 128,
                        'gamma': 0.95,
                        'learning_rate': 1e-3,
                        'gae_lambda': 0.95,  # lambda for Generalized Advantage Estimation
                        'rep_learning_rate': 1e-5,  # learning rate for learning next state representation
                        'seed': seed}

'''Parameters of Model'''
model_parameters = {'num_units': 64, 'seed': seed}
loss_coefficients = {'value': 0.5, 'entropy': 1e-2, 'representation': 0.0}

'''Write Parameters to log_file'''
if verbose:
    with open(log_file_name, "a") as f:
        f.write('Environment: {}, Frames: {} \n'.format(game, num_frames))
        f.write('Algorithm Parameters: {} \n'.format(algorithm_parameters))
        f.write('Model Parameters: {} \n'.format(model_parameters))
        f.write('Loss Coefficients: {} \n'.format(loss_coefficients))
        f.flush()

'''Initialize Environment & Model'''
env = Environment(game, seed)
num_actions = env.number_of_actions
state_space = env.state_space
model = Model(num_actions, state_space, model_parameters)
agent = A2C(model, num_actions, algorithm_parameters, loss_coefficients)

'''Train the Agent'''
reward_history, loss_history = agent.train(env, num_frames, logs, verbose)

'''Save Rewards and Losses'''
if verbose:
    np.save(reward_file_name, reward_history)
    np.save(loss_file_name, loss_history)


