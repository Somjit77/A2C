import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os


def running_mean(x, n):
    cumulative_sum = np.cumsum(np.insert(x, 0, 0))
    return (cumulative_sum[n:] - cumulative_sum[:-n]) / float(n)


def get_file(index):
    save_dir = "C:/Users/somjit/OneDrive/Codes/Model Based RL/Results/"  # Save Directory
    files_list = os.listdir(save_dir)
    with open(save_dir + files_list[index] + '/log.txt', 'r') as f:
        env_name = f.readline().split(',')[0].split(':')[1]  # Gets the environment name
    file_name = save_dir + files_list[index]  # Final files directory
    return env_name, file_name


def plot_reward(index, n):
    env_name, file_name = get_file(index)
    plt.figure()
    plt.title(env_name)
    rewards = np.load(file_name + '/rewards.npy')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.plot(running_mean(rewards, n))
    plt.savefig(file_name + '/rewards.pdf')
    plt.show()


def plot_losses(index):
    env_name, file_name = get_file(index)
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    ax1.set_title('Policy Loss')
    ax2.set_title('Value Loss')
    ax3.set_title('Reconstruction Loss')
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    f.tight_layout(pad=3.0)
    # Loss has 2 components organized as: Total, Policy, Value
    loss = np.load(file_name + '/loss.npy')
    reconstruction_loss = loss[:, 0]
    policy_loss = loss[:, 1]
    value_loss = loss[:, 2]
    plt.xlabel('Updates')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Loss')
    ax3.set_ylabel('Loss')
    ax1.plot(policy_loss, label='Policy Loss')
    ax2.plot(value_loss, label='Value Loss')
    ax3.plot(reconstruction_loss, label='Reconstruction Loss')
    plt.savefig(file_name + '/loss.pdf')
    plt.show()


def compare_algorithms(indices, label, n):
    env_name = []
    file_name = []
    for index in indices:
        env, file = get_file(index)
        env_name.append(env)
        file_name.append(file)
    # if not env_name.count(env_name[0]) == len(env_name):  # Check if all the environments are same
    #     raise Exception('Environments are different')
    plt.figure()
    plt.title(env_name[0])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    for index in range(len(indices)):
        rewards = np.load(file_name[index] + '/rewards.npy')
        plt.plot(running_mean(rewards, n), label=label[index])
    plt.legend()
    save_dir = "C:/Users/somjit/OneDrive/Codes/Model Based RL/Plots"
    try:
        plt.savefig(save_dir + '/rewards.pdf')
    except FileNotFoundError:
        os.makedirs("C:/Users/somjit/OneDrive/Codes/Model Based RL/Plots")
        plt.savefig(save_dir + '/rewards.pdf')
    plt.show()


if __name__ == "__main__":
    # plot_reward(-2, 10)
    # plot_losses(-1)
    compare_indices = [-1, -2]
    labels = [str(i) for i in compare_indices]
    labels = ['With Rep', 'Without Rep']
    compare_algorithms(compare_indices, labels, 50)
