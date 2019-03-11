import numpy as np
import scipy.ndimage
#import pyaudio
import time
import matplotlib
import keyboard
from pynput.keyboard import Key, Controller
#import cv2
import matplotlib.pyplot as plt
from mss import mss
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

dims = 3
percentile = 100

count = 40
audiosize = 0
pixels = None
channels = None
repeats = 1
base_input = 2
reward_size = 0
output_count = 1
max_freq = 50
randomsize = 0
video_count = base_input * repeats if pixels is None else pixels ** 2 * channels
audvis_count = video_count + audiosize

firing_history = 200
sensory_count = audvis_count + randomsize + reward_size
gamma = 0.002
threshold = 0.5
limits = 2
reward_amount = 12
decay = 1 - gamma / 100
video_stream = False
exp_decay = np.power(gamma, -np.arange(firing_history - 1))[None, :, None]


class Mind:
    def __init__(self, exec, init_gamma, mode=None):
        self.neurons = np.random.uniform(size=(count, dims))  # Physical position in space
        self.neurons[-1][-output_count:] = 0.9
        if pixels is not None:
            visual_map = np.mgrid[0.3:0.7:pixels * 1j, 0.3:0.7:pixels * 1j, 0.2:0.4:channels * 1j]
        else:
            visual_map = np.mgrid[0.3:0.7:base_input * 1j, 0.3:0.7:repeats * 1j, 0.29:0.31:1j]
        visual_map = visual_map.reshape(dims, -1).T
        self.mode = mode
        self.xor = np.zeros(video_count)
        self.neurons[:video_count] = visual_map
        self.dists = np.sqrt(np.mean(np.square(self.neurons[None, :, :] - self.neurons[:, None, :]), axis=-1))
        self.firings = np.random.binomial(size=(firing_history, count), n=1, p=0.5)
        self.connections = (
                (self.dists > 0) *
                (self.dists <= np.percentile(self.dists, percentile)) *
                np.random.uniform(size=self.dists.shape, low=-limits, high=limits)
        )

        # Disable connections coming into the sensory neurons
        self.connections[:, :sensory_count] = 0
        # # Disable direct connections from the inputs to the outputs
        self.connections[:sensory_count, -output_count:] = 0
        # Disable connections coming out of the sensory neurons
        # self.connections[:, -1] = 1
        # Disable intermediate loops:
        self.connections[sensory_count:, sensory_count:-output_count] = 0
        # # Disable connections coming out of the outputs
        # self.connections[-output_count:, :] = 0

        self.plastic = np.ones_like(self.connections)
        self.plastic[self.connections == 0] = 0
        self.gamma = init_gamma
        self.exec = exec
        self.lr_decay = 1 - self.gamma / 100
        self.acc_decay = 1 - self.gamma
        self.screen_cur = None
        self.screen_prev = None
        self.iter_num = 0
        self.reward = 0

        self.accumulation = np.ones_like(self.connections[sensory_count:-output_count, sensory_count:-output_count])
        self.accumulation *= 1 / (1 - self.acc_decay)
        self.up = np.zeros_like(self.firings[0, sensory_count:])
        self.upcount = 0
        self.down = np.zeros_like(self.firings[0, sensory_count:])
        self.downcount = 0
        self.sight = None
        self.sound = None

        self.performance = np.asarray([])

    def output(self, keyboard_to_press):
        # if self.firings[-1][-output_count:].mean() > self.firings[:, -output_count:].mean():
        # if np.random.binomial(1, self.firings[-1][-output_count:].mean()) > 0.5:
        if self.firings[-1][-output_count:].mean() >= 0.5:
            keyboard_to_press.press(" ")
        else:
            keyboard_to_press.release(" ")

    def fire(self):
        if self.sight is not None:
            visual = self.sight.flatten()
            # print("firings", np.floor((self.firings[-1] @ self.connections) * 10)/10)
            firings_next = ((self.firings[-1] @ self.connections) > self.firings.mean(0)).astype(float)
            # firings_next = ((self.firings[-1] @ self.connections) > threshold).astype(float)
            # firings_next[:len(visual)] = visual > visual.mean()
            if self.mode is "xor":
                firings_next[:len(visual)] = visual
                # print("a", firings_next)
            else:
                firings_next[:len(visual)] = visual / visual.max()
            if reward_size > 0:
                firings_next[audvis_count:audvis_count + reward_size] = self.reward
                # print("b", firings_next)
            # print(firings_next)
            if audiosize > 0:
                firings_next[len(visual):audvis_count] = self.sound > self.sound.mean()
            if randomsize > 0:
                firings_next[audvis_count:sensory_count] = np.random.binomial(size=(randomsize,), n=1, p=0.8)
            self.firings[:-1] = self.firings[1:]
            self.firings[-1] = firings_next

    def plot(self):
        plt.close()
        plt.imshow(np.concatenate((self.connections[:,:,None], self.connections[:,:,None], self.plastic[:,:,None]), axis=2))
        plt.savefig("dinoboi_" + str(self.iter_num) + ".png")
        plt.show()
        plt.close()

        if pixels is not None:
            plt.imshow(
                ((self.connections * self.firings.mean(0)).sum(1) /
                 self.firings.sum())[:video_count].reshape(pixels, pixels))
            plt.savefig("dinoboi_" + str(self.iter_num) + ".png")
            plt.show()
            plt.close()

    # Weaken old connections over time
    def decay(self):
        # if self.iter_num % 20000 == 0:
        #     self.plot()
        self.iter_num += 1
        self.connections *= decay

    def stdp(self, a, b):
        a = np.asarray(a)[:, None]
        b = np.asarray(b)[None, :]
        c = b - (1 - a)
        d = a * b
        e = np.multiply(c, d | d.T)
        return e

    def reinforce(self, alpha, hist=4):
        for i in range(hist):
            if alpha < 0:
                self.connections[:, self.firings[-i-1].astype(bool)] /= 1 + self.gamma * np.abs(alpha) / hist
            else:
                self.connections[:, self.firings[-i-1].astype(bool)] *= 1 + self.gamma * np.abs(alpha) / hist

    def learn(self, alpha=1.0):
        # print("Alpha", alpha)
        # thought    = self.firings[:, sensory_count:-output_count]
        # print("STDP", self.stdp(self.firings[-2], self.firings[-1]), self.plastic)
        wow = np.multiply(np.multiply(self.stdp(self.firings[-2], self.firings[-1]), self.plastic), self.connections)
        # wow += np.multiply(np.multiply(self.stdp(self.firings[-3], self.firings[-2]), self.plastic), self.connections) * 0.3
        # wow += np.multiply(np.multiply(self.stdp(self.firings[-4], self.firings[-3]), self.plastic), self.connections) * 0.09
        # print("WOW", np.abs(wow).mean())
        #plt.imshow(wow)
        #plt.show()

        # self.accumulation += np.abs(wow[sensory_count:-output_count, sensory_count:-output_count] * self.gamma * alpha)
        # self.accumulation *= self.acc_decay
        # subplastic = self.plastic[sensory_count:-output_count, sensory_count:-output_count]
        # synapse_strength = np.abs(self.connections)[sensory_count:-output_count, sensory_count:-output_count]
        # print(self.accumulation)
        # updates = (self.accumulation <= 0.0001/(1 - self.acc_decay)) * (synapse_strength <= np.sum(synapse_strength * subplastic) / subplastic.sum())
        # self.plastic[sensory_count:-output_count, sensory_count:-output_count][updates] = 0
        # print("Plastic", self.plastic.mean())
        # print("Wow", wow)
        self.connections += self.connections * wow * self.gamma * alpha # * wow.sum() / 10
        self.connections = self.connections.clip(-limits, limits)
        # print(self.connections)
        self.gamma *= self.lr_decay

    def visualize(self):
        for neuron in range(-100,-90):
            try:
                sumer_max = np.zeros((pixels, pixels, channels))
                sumer_min = np.zeros((pixels, pixels, channels))
                sumer_diff = np.zeros((pixels, pixels, channels))
                for _ in range(200):
                    maxer = (-1, None)
                    miner = (10000, None)
                    initimage_max = np.zeros((pixels, pixels, channels))
                    initimage_min = np.zeros((pixels, pixels, channels))
                    for count in range(6):
                        for _ in range(100):
                            image_max = (1 - 1 / np.power(1.2, count)) * initimage_max + 1 / np.power(1.2, count) * np.random.uniform(size=(pixels, pixels, channels))
                            image_min = (1 - 1 / np.power(1.2, count)) * initimage_min + 1 / np.power(1.2, count) * np.random.uniform(size=(pixels, pixels, channels))
                            fire_max = np.zeros_like(self.firings[-1])
                            fire_min = np.zeros_like(self.firings[-1])
                            for c in range(3):
                                # fire[:pixels ** 2 * channels] = (image > image.mean()).flatten()
                                fire_max[:pixels ** 2 * channels] = image_max.flatten()
                                fire_min[:pixels ** 2 * channels] = image_min.flatten()
                                fire_max = ((fire_max @ self.connections) > threshold).astype(float)
                                fire_min = ((fire_min @ self.connections) > threshold).astype(float)
                            fire_max = fire_max @ self.connections
                            fire_min = fire_min @ self.connections
                            if fire_max[neuron] > maxer[0]:
                                maxer = (fire_max[neuron], image_max)
                            if fire_min[neuron] < miner[0]:
                                miner = (fire_min[neuron], image_min)
                        initimage_max = maxer[1]
                        initimage_min = miner[1]
                    # plt.imshow(((maxer[1] * 255).clip(0, 255)).astype(int))
                    # plt.plot(self.neurons[neuron][0] * (pixels - 1), self.neurons[neuron][1] * (pixels - 1), 'bo')
                    # plt.show()
                    # plt.imshow(((miner[1] * 255).clip(0, 255)).astype(int))
                    # plt.plot(self.neurons[neuron][0] * (pixels - 1), self.neurons[neuron][1] * (pixels - 1), 'bo')
                    # plt.show()
                    diff = maxer[1] - miner[1]
                    sumer_diff += diff
                    sumer_max += maxer[1]
                    sumer_min += miner[1]
                    # plt.imshow((diff - diff.min())/(diff.max() - diff.min()))
                    # plt.plot(self.neurons[neuron][0] * (pixels - 1), self.neurons[neuron][1] * (pixels - 1), 'bo')
                    # plt.title(str(neuron))
                    # plt.show()
                    print("----")
                plt.imshow((sumer_max - sumer_max.min()) / (sumer_max.max() - sumer_max.min()))
                plt.plot(self.neurons[neuron][0] * pixels - 0.5, self.neurons[neuron][1] * pixels - 0.5, 'bo')
                plt.title("Max" + str(neuron))
                plt.show()

                plt.imshow((sumer_min - sumer_min.min()) / (sumer_min.max() - sumer_min.min()))
                plt.plot(self.neurons[neuron][0] * pixels - 0.5, self.neurons[neuron][1] * pixels - 0.5, 'bo')
                plt.title("Min" + str(neuron))
                plt.show()

                plt.imshow((sumer_diff - sumer_diff.min()) / (sumer_diff.max() - sumer_diff.min()))
                plt.plot(self.neurons[neuron][0] * pixels - 0.5, self.neurons[neuron][1] * pixels - 0.5, 'bo')
                plt.title("Diff" + str(neuron))
                plt.show()
            except:
                pass

    def screenshot(self):
        sct = mss()
        im = sct.grab({"top":100 ,"left":0, "width":500, "height":230})
        rgb = Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")
        visual = rgb.resize(size=(pixels, pixels), resample=Image.BICUBIC)
        visual = np.array(visual)
        # plt.imshow(visual)
        # plt.show()
        return visual

    def audiovision(self, cam, mic=None):
        if audiosize > 0:
            p = pyaudio.PyAudio()
            mic = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=audiosize)
            data = mic.read(audiosize)
            self.sound = np.fromstring(data, dtype=np.int16)
        else:
            self.sound = None

        if cam is not None:
            ret, frame = cam.read()
            base = cv2.resize(frame, dsize=(pixels, pixels), interpolation=cv2.INTER_AREA)
            self.sight = base.mean(-1)[:, :, None]
        elif self.mode == "screen":
            self.sight = self.screenshot()
            self.screen_prev = self.screen_cur
            self.screen_cur = self.sight.copy()
        elif self.mode == "xor":
            self.sight = self.xor

def main():
    def input():
        george.audiovision(cam)

    def output(keyboard):
        george.output(keyboard)

    def processing(count):
        george.fire()
        if george.mode == "xor":
            if count > 2 and count % 20 > 4:
                if george.xor.sum() / repeats % 2 == george.firings[-1][-output_count:].mean().round():
                    george.performance = np.append(george.performance, 1)
                    george.reward = 1
                    if george.xor.sum() / repeats % 2 == 1: # True positive
                        george.learn(1)
                    else: # True negative
                        george.learn(1)
                else:
                    george.performance = np.append(george.performance, 0)
                    george.reward = 0
                    if george.xor.sum() / repeats % 2 == 1: # False positive
                        george.reinforce(-reward_amount)
                        george.learn(0.5)
                    else: # False negative
                        george.reinforce(reward_amount)
                        george.learn(0.5)
        if george.mode == "dino":
            if george.screen_prev is not None:
                nov = np.abs(george.screen_cur - george.screen_prev).mean()
            else:
                nov = 1
            if nov > 0.0001:
                george.learn(1)
            else:
                george.learn(-0.1)
        george.decay()
        if (count % n == n - 1):
            george.visualize()

    def show():
        if george.sight is not None:
            plt.imshow(george.sight)
            plt.show()

    if video_stream:
        cam = cv2.VideoCapture(0)
        cam.set(3, 36)
        cam.set(4, 64)
    else:
        cam = None

    counts = 40
    total = np.zeros(counts)
    for cur in range(counts):
        n = 100000
        executor = ThreadPoolExecutor(max_workers=3)
        george = Mind(executor, gamma, mode="xor")
        keyboard_press = Controller()
        for count in range(30000):
            # show()
            input()
            count += 1
            if count % 20 == 0:
                george.xor = np.tile(np.random.binomial(1, 0.5, (2,)), repeats)
            processing(count)
            if george.mode is not "xor":
                output(keyboard_press)
            # time.sleep(1/max_freq)
            # if keyboard.is_pressed("space"): #if key space is pressed.You can also use right,left,up,down and others like a,b,c,etc.
            #     print("Rewarded")
            #     george.reward()
        print(george.performance[-4000:].mean())
        if george.performance[-4000:].mean() > 0.90:
            george.plot()
        total[cur] = george.performance[-4000:].mean() > 0.8
    print(total.mean(), total.std())
main()
