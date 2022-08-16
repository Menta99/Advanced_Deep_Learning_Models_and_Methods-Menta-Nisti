from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle


if __name__ == '__main__':
    sns.set(rc={'figure.figsize': (16, 9)})
    algorithm = 'Magic'
    environment = 'ConnectFour'
    representation = 'Graphic'
    opponent = 'MinMaxRandom'
    agent_turn = 'Random'

    config_name = algorithm + '_' + environment + '_' + representation + '_' + opponent + '_' + agent_turn
    data_path = '../Results/' + config_name + '/scores.pkl'
    f = open(data_path, 'rb')
    results_random = pickle.load(f)
    f.close()
    data_random_reward = pd.DataFrame(np.array([(key, value[i][0]) for key, value in results_random.items() for i in
                                                range(len(value))]), columns=['episode', 'reward'])
    data_random_reward['rolling'] = data_random_reward.reward.rolling(100).mean()
    data_random_length = pd.DataFrame(np.array([(key, value[i][1]) for key, value in results_random.items() for i in
                                                range(len(value))]), columns=['episode', 'length'])
    data_random_length['rolling'] = data_random_length.length.rolling(100).mean()

    fig, ax = plt.subplots()
    ax.set(ylim=(-2, 1))
    ax = sns.lineplot(x='episode', y='reward', data=data_random_reward, label='random')
    ax = sns.lineplot(x='episode', y='rolling', data=data_random_reward, label='random_rolling')
    plt.show()

    fig, ax = plt.subplots()
    ax.set(ylim=(0, 42))
    ax = sns.lineplot(x='episode', y='length', data=data_random_length, label='random')
    ax = sns.lineplot(x='episode', y='rolling', data=data_random_length, label='random_rolling')
    plt.show()
