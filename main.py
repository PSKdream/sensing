import os
from environment import Environment
import pandas as pd
from algorithms.deep_q_learning import DeepQLearningAgent
from common import get_initial_state_variables
import time

def load_dataset(path):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, path) #5200 rows
    #file_path = os.path.join(script_dir, './dataset/real_data_trace.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/single_channel_no_prob_strange_order.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/single_channel_with_switching_prob.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/multiple_channels_with_switching_prob.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/multiple_channels_no_prob.csv') #5200 rows

    df = pd.read_csv(file_path)

    df_train = df[df.index <= 4599]
    df_test = df[df.index > 4599]

    return df_train, df_test

def main():
    #load dataset, divide into train and test
    df_train, df_test = load_dataset('./dataser_Coor/SU1/perfectly_correlated.csv')
    df_train2, df_test2 = load_dataset('./dataser_Coor/SU2/perfectly_correlated.csv')
    df_train3, df_test3 = load_dataset('./dataser_Coor/SU3/perfectly_correlated.csv')
    df_train4, df_test4 = load_dataset('./dataser_Coor/SU4/perfectly_correlated.csv')
    df_train5, df_test5 = load_dataset('./dataser_Coor/SU5/perfectly_correlated.csv')


    #environment should have the entire dataset as an input parameter, but train and test methods
    environment = Environment()
    agent = DeepQLearningAgent(environment)

    #n_episodes = 800
    n_episodes = 800
    episode_length = 200
    ####n_episodes = 25
    print('agent training started')
    t1 = time.time()
    #agent.train5(df_train,df_train2,df_train3,df_train4,df_train5, n_episodes, episode_length)
    agent.train(df_train, n_episodes, episode_length)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    print ('Test on the test dataset')
    agent.test(df_test)


if __name__ == '__main__':
  main()
