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
from wheatley import firing_history, repeats, output_n, total_n, Mind

mode = "gym" # What game are you using
# env_name = 'MountainCar-v0'
# env_name = 'MountainCarContinuous-v0'
# env_name = 'CartPole-v0'
env_name = 'Pendulum-v0'
# env_name = 'HalfCheetah-v1'
# env_name = 'Tutankham-ram-v0'

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

            if env_name == 'Pendulum-v0':
                # action = (((wheatley.firings[-2] @ wheatley.connections)[-output_n:].mean()))
                # action = -1.5 if wheatley.firings[-1][-output_n:].mean() > wheatley.firings[:, -output_n:].mean() else 1.5
                action = 8 * (-0.5 + (wheatley.firings[-1, -output_n:].mean(0) > wheatley.firings[:, -output_n:].mean(0)).astype(float).mean())
                # action = 2 * (-0.5 + (wheatley.firings[-1][-output_n:].mean(0) > wheatley.firings[:, -output_n:].mean(0)).mean())
                # print(action)
            elif env_name == 'MountainCar-v0':
                action = 2 if wheatley.firings[-1][-output_n:].mean() > wheatley.firings[:, -output_n:].mean() else 0
            elif env_name == 'MountainCarContinuous-v0':
                # action = 2 * ((wheatley.firings[-2] @ wheatley.connections)[-output_n:].mean())
                action = 2 * (-0.5 + (wheatley.firings[-1][-output_n:].mean(0) > wheatley.firings[:, -output_n:].mean(0)).mean())
            elif env_name == 'CartPole-v0':
                # print((wheatley.firings[-2] @ wheatley.connections)[-env.action_space.n:])
                # action = (wheatley.firings[-2] @ wheatley.connections)[-env.action_space.n:].argmax()
                action = 1 if wheatley.firings[-1][-output_n:].mean() > wheatley.firings[:, -output_n:].mean() else 0
                # action = 1 if wheatley.firings[-1][-output_n:].mean() > wheatley.firings[:, -output_n:].mean() else 0
            elif env_name == 'HalfCheetah-v1':
                action = 4 * (-0.5 + (wheatley.firings[-1][-output_n:].mean(0) > wheatley.firings[:, -output_n:].mean(0)).mean())
            else:
                action = (wheatley.firings[-1][-output_n:].mean(0) > wheatley.firings[:, -output_n:].mean(0)).mean()
                if discrete:
                    action = (wheatley.firings[-2] @ wheatley.connections)[-env.action_space.n:].argmax()
                    # action *= env.action_space.n - 1
                    # action = action.round().astype(int)
                else:
                    action -= 0.5
                    action *= env.action_space.high - env.action_space.low

            # print(env.action_space.high - env.action_space.low)

            # print(action)
            if not discrete:
                action = [action]

            observation, reward, done, info = env.step(action)
            wheatley.sight = 2 * (-0.5 + (observation - env.observation_space.low) /
                                  (env.observation_space.high - env.observation_space.low))
            wheatley.total_reward += reward
            # print(reward)

            # outputer = True
            outputer = False
            if outputer:
                print("----")
            if env_name == 'CartPole-v0':
                if wheatley.expected_reward == None:
                    wheatley.expected_reward = 10.01
                # wheatley.reinforce((count / 10 - 1) * wheatley.gamma / wheatley.init_gamma, hist=count, printer=outputer)
                wheatley.reinforce(1 * nov.mean() * wheatley.gamma / wheatley.init_gamma, hist=10, printer=outputer)
                # wheatley.learn(0.01, printer=outputer)
                if done:
                    p = 1 / 100
                    wheatley.expected_reward = (1 - p) * wheatley.expected_reward + p * count
                    print(wheatley.expected_reward)
                    if count < 100:
                        wheatley.reinforce(-0.1 / (count - wheatley.expected_reward), hist=10)
                        return count
                    wheatley.reinforce(1, hist=count, printer=outputer)
                    return count
            elif env_name == 'MountainCar-v0':
                wheatley.learn(0.1)
                wheatley.reinforce(10 * nov.mean() * wheatley.gamma / wheatley.init_gamma, hist=50)
                if observation[0] >= 0.5:
                    wheatley.reinforce(0, hist=count)
                    return count
                if count == 1000:
                    wheatley.reinforce(-0.1, hist=count)
                    return count
            elif env_name == 'MountainCarContinuous-v0':
                wheatley.learn(0.1, printer=outputer)
                wheatley.reinforce(nov.mean() * wheatley.gamma / wheatley.init_gamma, hist=50, printer=outputer)
                if observation[0] >= 0.5:
                    wheatley.reinforce(10 * 1000/count, hist=count, printer=outputer)
                    return count
                if count == 1000:
                    wheatley.reinforce(-0.1, hist=count, printer=outputer)
                    return count
            elif env_name == 'Pendulum-v0':
                # if reward > 0:
                #     wheatley.reinforce(3, hist=50)
                # else:
                #     wheatley.reinforce(reward / 10, hist=50)
                if wheatley.expected_reward == None:
                    # wheatley.expected_reward = reward - 5
                    wheatley.expected_reward = -7
                if wheatley.expected_novelty == None:
                    wheatley.expected_novelty = 0
                wheatley.learn((reward - wheatley.expected_reward) / 10, printer=outputer)
                # wheatley.reinforce(2 * nov.mean() * wheatley.gamma / wheatley.init_gamma, hist=count, printer=outputer)
                # wheatley.reinforce((reward - wheatley.expected_reward) * wheatley.gamma / wheatley.init_gamma, hist=count, printer=outputer)
                # p = 1 / 5000
                # wheatley.expected_reward = (1 - p) * wheatley.expected_reward + p * reward
                # wheatley.expected_novelty = (1 - p) * wheatley.expected_reward + p * nov.mean()
                if count == 1000:
                    # outputer = True
                    wheatley.reinforce((wheatley.total_reward - wheatley.expected_reward * count) / 10, hist=count, printer=outputer)
                    return wheatley.total_reward
            elif env_name == 'HalfCheetah-v1':
                if wheatley.expected_reward == None:
                    wheatley.expected_reward = reward
                wheatley.reinforce(np.abs(nov).mean() * wheatley.gamma / wheatley.init_gamma, hist=50)
                wheatley.reinforce((reward - wheatley.expected_reward), hist=count)
                wheatley.learn(reward / 10)
                if count == 200:
                    wheatley.reinforce(-5, hist=count)
                    return wheatley.total_reward
                if done:
                    wheatley.reinforce(1000/count, hist=count)
                    return wheatley.total_reward
            else:
                if wheatley.expected_reward == None:
                    wheatley.expected_reward = reward
                if wheatley.expected_novelty == None:
                    wheatley.expected_novelty = nov.mean()
                # print(np.abs(nov).mean())
                p = 1 / 200
                wheatley.expected_reward = (1 - p) * wheatley.expected_reward + p * reward
                wheatley.expected_novelty = (1 - p) * wheatley.expected_reward + p * nov.mean()
                wheatley.reinforce((-wheatley.expected_novelty + nov.mean()) * wheatley.gamma / wheatley.init_gamma, hist=50)
                # print((reward - wheatley.expected_reward))
                # wheatley.reinforce(0.01 * (reward - wheatley.expected_reward), hist=count)
                wheatley.learn(50)
                if count == 500:
                    # wheatley.reinforce(-5, hist=count)
                    return wheatley.total_reward
                # if done:
                #     wheatley.reinforce(1000/count, hist=count)
                #     return wheatley.total_reward



            # if env_name == 'Pendulum-v0':
            #     if wheatley.expected_reward == None:
            #         wheatley.expected_reward = reward
            #     wheatley.reinforce(np.abs(nov).mean() * wheatley.gamma / wheatley.init_gamma, hist=50)
            #     wheatley.reinforce(1 * (reward - wheatley.expected_reward), hist=count)
            #     wheatley.learn(0.1)
            #     p = 1 / 200
            #     wheatley.expected_reward = (1 - p) * wheatley.expected_reward + p * reward
            #     if done:
            #         return wheatley.total_reward

        wheatley.decay()
        if (count % n == n - 1):
            wheatley.visualize()

    def show():
        if wheatley.sight is not None:
            vis = np.concatenate((wheatley.sight, wheatley.sight, wheatley.sight), -1)
            plt.imshow(vis)
            plt.show()



    counts = 10000
    total = np.zeros(counts)
    threader = ThreadPoolExecutor(max_workers=3)

    if mode == "gym":
        env = gym.make(env_name)
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        # wheatley = Mind(threader, mode=mode, base_n=len(env.observation_space.high), lr_decay=0.9999)
        wheatley = Mind(threader, mode=mode, base_n=len(env.observation_space.high), gamma=0.001, lr_decay=0.9999)
    else:
        wheatley = Mind(threader, mode=mode, base_n=5)
    if wheatley.video_stream:
        cam = cv2.VideoCapture(0)
        cam.set(3, 36)
        cam.set(4, 64)
    else:
        cam = None

    iter_counts = []
    for cur in range(counts):
        observation = env.reset()
        wheatley.total_reward = 0
        n = 100000
        keyboard_press = Controller()
        for step in range(1000000):
            if cur % 20 == 0:
                env.render()
                # if step == 0:
                #     print("CUR", cur)
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

from gym import envs
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

for envo in env_ids[-100:]:
    # env_name = envo
    main()
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     quit()
    # except Exception as e:
    #     print(e)
    #     continue
