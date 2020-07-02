import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Suppress Tensorflow Messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CPU/GPU
import numpy as np
from agent import A2C
from env import Environment

'''Log directory'''
verbose = True
if verbose:
    save_dir = os.path.join(os.getcwd() + '/Results', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_dir)
    log_file_name = save_dir + '/log.txt'
    reward_file_name = save_dir + '/rewards'
    loss_file_name = save_dir + '/loss'
else:
    log_file_name = ""

'''Environment Parameters'''
game = 'MiniGrid-DoorKey-6x6-v0'
seed = 0  # Seed for Env, TF, Numpy
num_frames = 2e6  # Million Frames
logs = {'log_interval': 50,  # Number of Episodes after which to print output/save batch output
        'log_file_name': log_file_name
        }

'''Parameters of Algorithm'''
algorithm_params = {'batch_size': 256,
                    'gamma': 0.95,
                    'learning_rate': 7e-4,
                    'gae_lambda': 0.95,  # lambda for Generalized Advantage Estimation
                    'seed': seed
                    }
model_params = {'use_model': 1,  # Using Models
                'rep_learning_rate': 1e-3,  # learning rate for learning next state representation
                'reward_learning_rate': 1e-3,  # learning rate for learning reward
                'model_critic_learning_rate': 1e-5,  # learning rate for critic in model-based rollouts
                'buffer_size': 10000,
                'planning_steps': 10,
                'rollouts': 1,
                }

'''Parameters of Model'''
architecture_params = {'num_units': 64}
loss_coefficients = {'value': 0.5, 'entropy': 1e-2}

'''Write Parameters to log_file'''
if verbose:
    with open(log_file_name, "a") as f:
        f.write('Environment: {}, Frames: {} \n'.format(game, num_frames))
        f.write('Algorithm Parameters: {} \n'.format(algorithm_params))
        f.write('Model Parameters: {} \n'.format(model_params))
        f.write('Loss Coefficients: {} \n'.format(loss_coefficients))
        f.flush()

'''Initialize Environment & Model'''
env = Environment(game, seed)
num_actions = env.number_of_actions
state_space = env.state_space
agent = A2C(num_actions, state_space, architecture_params, model_params, algorithm_params, loss_coefficients)

'''Train the Agent'''
reward_history, loss_history = agent.train(env, num_frames, logs, verbose)

'''Save Rewards and Losses'''
if verbose:
    np.save(reward_file_name, reward_history)
    np.save(loss_file_name, loss_history)
