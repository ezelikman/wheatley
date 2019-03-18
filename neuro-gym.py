import numpy as np
import scipy.ndimage
import time
import matplotlib
import keyboard
import gym
#import cv2
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plt
from mss import mss
from concurrent.futures import ThreadPoolExecutor
from wheatley import init_gamma, firing_history, repeats, output_n, Mind

mode = "gym" # What game are you using
# env_name = 'MountainCar-v0'
# env_name = 'CartPole-v0'
env_name = 'Pendulum-v0'

max_freq = 50  # Maximum per neuron firing frequency per second, less processor-dependent
reward_amount = 12  # How much to update the connections in response to global rewards

def main():
    def input():
        wheatley.audiovision(cam)

    def output(keyboard):
        wheatley.output(keyboard)

    def processing(count, env=None):
        wheatley.fire()
        if wheatley.mode == "xor":
            if count % 20 > 4:
                if wheatley.xor.sum() / repeats % 2 == wheatley.firings[-1][-output_n:].mean().round():
                    wheatley.performance = np.append(wheatley.performance, 1)
                    wheatley.reward = 1
                    if wheatley.xor.sum() / repeats % 2 == 1:  # True positive
                        wheatley.learn(1)
                    else:  # True negative
                        wheatley.learn(1)
                else:
                    wheatley.performance = np.append(wheatley.performance, 0)
                    wheatley.reward = 0
                    if wheatley.xor.sum() / repeats % 2 == 1: # False positive
                        wheatley.reinforce(-reward_amount)
                        wheatley.learn(0.5)
                    else: # False negative
                        wheatley.reinforce(reward_amount)
                        wheatley.learn(0.5)

        if wheatley.mode == "dino":
            if wheatley.screen_prev is not None:
                nov = np.abs(wheatley.screen_cur - wheatley.screen_prev).mean()
            else:
                nov = 1
            if nov > 0.0001:
                wheatley.learn(1)
            else:
                wheatley.reinforce(-reward_amount)
                wheatley.learn(0.5)

        if wheatley.mode == "gym":
            nov = np.abs(np.multiply(np.multiply(
                wheatley.stdp(wheatley.firings[-2], wheatley.firings[-1]), wheatley.plastic
            ), wheatley.connections))
            # print("Nov", np.abs(nov).mean())
            action = ((wheatley.firings[-2] @ wheatley.connections)[-1])

            if env_name == 'MountainCar-v0':
                action = 2 if wheatley.firings[-1][-output_n:].mean() > wheatley.firings[:, -output_n:].mean() else 0
            if env_name == 'CartPole-v0':
                action = 1 if wheatley.firings[-1][-output_n:].mean() > wheatley.firings[:, -output_n:].mean() else 0

            if not discrete:
                action = [action]
            observation, reward, done, info = env.step(action)
            wheatley.sight = 2 * (-0.5 + (observation - env.observation_space.low) /
                                  (env.observation_space.high - env.observation_space.low))
            wheatley.total_reward += reward
            # print(reward)

            if env_name == 'CartPole-v0':
                if done:
                    if count < 200:
                        print("Bad")
                        # wheatley.reinforce(count / 10, hist=count)
                        wheatley.reinforce(-10 / count, hist=count)
                    else:
                        wheatley.reinforce(1)
                if done:
                    print("Episode finished after {} timesteps".format(count+1))
                    return True

            if env_name == 'MountainCar-v0':
                wheatley.reinforce(np.abs(nov).mean() * wheatley.gamma / init_gamma, hist=50)
                wheatley.learn(0.1)
                if observation[0] >= 0.5:
                    wheatley.reinforce(1000/count, hist=count)
                    return count
                if count == 1000:
                    wheatley.reinforce(-5, hist=count)
                    return count

            if env_name == 'Pendulum-v0':
                wheatley.reinforce(reward - wheatley.expected_reward, hist=count)
                wheatley.expected_reward = 0.9 * wheatley.expected_reward + 0.1 * reward
                # wheatley.learn(0.2)
                if done:
                    return wheatley.total_reward

        wheatley.decay()
        if (count % n == n - 1):
            wheatley.visualize()

    def show():
        if wheatley.sight is not None:
            vis = np.concatenate((wheatley.sight, wheatley.sight, wheatley.sight), -1)
            plt.imshow(vis)
            plt.show()



    counts = 200
    total = np.zeros(counts)
    threader = ThreadPoolExecutor(max_workers=3)

    if mode == "gym":
        env = gym.make(env_name)
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        wheatley = Mind(threader, mode=mode, base_n=len(env.observation_space.high))
    else:
        wheatley = Mind(threader, mode=mode, base_n=5)

    if wheatley.video_stream:
        cam = cv2.VideoCapture(0)
        cam.set(3, 36)
        cam.set(4, 64)
    else:
        cam = None

    iter_counts = []
    wheatley.expected_reward = 0
    for cur in range(counts):
        observation = env.reset()
        wheatley.total_reward = 0
        n = 100000
        keyboard_press = Controller()
        for step in range(1000000):
            if cur % 40 == 0:
                env.render()
            # print(observation)
            # show()
            input()
            if step % 20 == 0:
                wheatley.xor = np.tile(np.random.binomial(1, 0.5, (2,)), repeats)
            done = processing(step, env)
            # time.sleep(0.1)
            if done != None:
                print(done)
                iter_counts.append(done)
                break
            if wheatley.mode == "dino":
                output(keyboard_press)
    print(iter_counts)
            # time.sleep(1/max_freq)
    print(total.mean(), total.std())

main()
