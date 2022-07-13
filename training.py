import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys

from tensorflow.keras.optimizers import Adam
import model as md


def readDataset(path):
    """Read videos from the given path for training

    Args:
        path (string): path to directory containing videos of a single action

    Returns:
        nparray: video dataset
    """
    videos = []
    for video_folder_path in os.listdir(path):
        video = path + '/' + video_folder_path
        frames = []
        image_files = os.listdir(video)
        image_files.sort()
        for frame in image_files:
            frame_path = video + '/' + frame
            frame = Image.open(frame_path)
            temp_frame_np = np.asarray(frame)
            frame_np = np.zeros(temp_frame_np.shape, dtype=np.uint8)
            for i in range(frame_np.shape[0]):
                for j in range(frame_np.shape[1]):
                    if temp_frame_np[i][j] >= 200:
                        frame_np[i][j] = 1
                    else:
                        frame_np[i][j] = 0
            frames.append(frame_np)
        videos.append(np.stack(frames))
    dataset = np.clip(np.stack(videos), 0, 255)

    return dataset


paths = [
    'dataset/walking_images_straight_hands/',
    'dataset/walking_images_r_2_l_straight_hands',
    'dataset/hand_raising_images', 'dataset/jumping_images/'
]
condition_matrices = [
    2 * np.ones((32, 32)),
    np.ones((32, 32)), -1 * np.ones((32, 32)), -2 * np.ones((32, 32))
]

dataset_videos = []
dataset_labels = []

for path, condition_matrix in zip(paths, condition_matrices):
    dataset = readDataset(path)
    for video in dataset:
        dataset_videos.append(video)
        dataset_labels.append(condition_matrix)

dataset_videos = np.asarray(dataset_videos)
dataset_labels = np.asarray(dataset_labels)

env = md.Environment(dataset_videos, dataset_labels)
optimizer = Adam(learning_rate=0.01)
agent = md.Agent(env, optimizer, 1024)

batch_size = 32
num_of_episodes = 10000
timesteps_per_episode = 10
rewards = []

for episode in range(num_of_episodes):
    env.reset()
    last_frame, condition_matrix = env.get_complete_current_state()
    reward = 0
    avg_rwrd = 0
    terminated = False

    epsilon = 0.99 * ((1.03)**(-episode)) + 0.01

    for timestep in range(timesteps_per_episode):
        act_randomly = np.random.rand() <= epsilon
        action, scaled_image, q_values = agent.act([last_frame, condition_matrix],
                                                   act_randomly)
        action = np.asarray(action, dtype=np.int8)
        reward, next_state, terminated, ground_truth = env.act(action)
        agent.store(scaled_image, action, reward, next_state, terminated,
                    condition_matrix)
        reward_sum = np.sum(reward)
        avg_rwrd += reward_sum

        if terminated:
            agent.align_target_model()
        elif len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)

        last_frame = next_state

        if terminated:
            break

    avg_rwrd /= 8

    if (episode + 1) % 1 == 0:
        rewards.append(avg_rwrd)
        print("Episode: {}".format(episode + 1))
        print(f"Reward: {avg_rwrd}")
    if (episode + 1) % 5 == 0:
        plt.plot(list(range(len(rewards))), rewards)
        plt.savefig(f'./plt.png')
        plt.close()
    agent.save('./trained_agent')
