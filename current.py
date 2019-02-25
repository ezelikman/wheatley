import numpy as np
import scipy.ndimage
import pyaudio
import time
import matplotlib
import keyboard
from pynput.keyboard import Key, Controller
import cv2
import matplotlib.pyplot as plt
from mss import mss
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


class Mind:
    def __init__(self, exec, init_gamma):
        self.neurons = np.random.uniform(size=(count, dims))
        visual_map = np.mgrid[0.3:0.7:pixels * 1j, 0.3:0.7:pixels * 1j, 0.49:0.51:channels * 1j]
        visual_map = visual_map.reshape(dims, -1).T
        self.neurons[:pixels ** 2 * channels] = visual_map
        self.dists = np.sqrt(np.mean(np.square(self.neurons[None, :, :] - self.neurons[:, None, :]), axis=-1))
        self.firings = np.random.binomial(size=(firing_history, count), n=1, p=0.5)
        self.connections = (
                (self.dists > 0) *
                (self.dists < np.percentile(self.dists, percentile)) *
                np.random.uniform(size=self.dists.shape, low=-2, high=2)
        )
        self.plastic = np.ones_like(self.connections)
        # Disable connections coming into the sensory neurons
        self.connections[:, :sensory_count] = 0
        self.plastic[self.connections == 0] = 0
        # Disable connections coming out of the sensory neurons
        # self.connections[-output_count:, :] = 0
        self.gamma = init_gamma
        self.exec = exec
        self.lr_decay = 0.999999
        self.acc_decay = 0.9

        self.accumulation = np.ones_like(self.connections[sensory_count:-output_count, sensory_count:-output_count])
        self.accumulation *= 1 / (1 - self.acc_decay)
        self.up = np.zeros_like(self.firings[0, sensory_count:])
        self.upcount = 0
        self.down = np.zeros_like(self.firings[0, sensory_count:])
        self.downcount = 0
        self.sight = None
        self.sound = None

    def output(self, keyboard_to_press):
        # print(self.firings[-1][-output_count:].mean(), "Out")
        # if self.firings[-1][-output_count:].mean() > 0.5:
        #     keyboard_to_press.press(" ")
        if self.firings[-1][-output_count:].mean() > self.firings[:, -output_count:].mean():
        # if self.firings[-1][-output_count:].mean () >= 0.5:
            keyboard_to_press.press(" ")
        else:
            keyboard_to_press.release(" ")


    def reward(self):
        thought = self.firings[:, sensory_count:-output_count]
        nov = np.abs(thought[-1] - thought[-10]).mean() / thought.sum() * 2E5
        print("novelty", nov)
        # self.connections[:, self.firings[-1]] *= reward_amount * (1 + nov)
        # self.connections = self.connections.clip(-limits, limits)
        # alpha = 10 if int(nov) > 0 else -10
        self.learn(nov)


    def fire(self):
        if self.sight is not None:
            visual = self.sight.flatten()
            firings_next = ((self.firings[-1] @ self.connections) > threshold).astype(float)
            # firings_next[:len(visual)] = visual > visual.mean()
            firings_next[:len(visual)] = visual / visual.max()
            # print(firings_next)
            if audiosize > 0:
                firings_next[len(visual):audvis_count] = self.sound > self.sound.mean()
            if randomsize > 0:
                firings_next[audvis_count:sensory_count] = np.random.uniform(randomsize)
            self.firings[:-1] = self.firings[1:]
            self.firings[-1] = firings_next
            # if (int(time.time()) % 10 != 0):
            #     if (int(time.time() / 10) % 2 == 0):
            #         print("Action One", int(time.time() - base_time), end=" ")
            #         self.up += firings_next[sensory_count:].astype(int)
            #         s  elf.upcount += 1
            #     else:
            #         print("Action Two", int(time.time() - base_time), end=" ")
            #         self.down += firings_next[sensory_count:].astype(int)
            #         self.downcount += 1
            # else:
            #     print("Switch Now", int(time.time() - base_time), end=" ")
            # print(self.firings[-1][-output_count:], np.max(np.abs(self.down / self.downcount - self.up / self.upcount)) )

    # Weaken old connections over time
    def decay(self):
        self.connections *= decay

    # Identify how well it correlated
    def learn(self, alpha=1):
        # print("Alpha", alpha)
        # thought    = self.firings[:, sensory_count:-output_count]
        wow = (self.firings[-1][None, :] - self.firings[-1][:, None]) * self.plastic
        self.accumulation += np.abs(wow[sensory_count:-output_count, sensory_count:-output_count]) \
                             * self.gamma * alpha
        self.accumulation *= self.acc_decay
        subplastic = self.plastic[sensory_count:-output_count, sensory_count:-output_count]
        synapse_strength = np.abs(self.connections)[sensory_count:-output_count, sensory_count:-output_count]
        updates = (self.accumulation <= 0.3/(1 - self.acc_decay)) * (synapse_strength >= np.sum(synapse_strength * subplastic) / subplastic.sum())
        subplastic[updates] = 0
        # print("Plastic", np.mean(self.plastic))
        self.connections += self.connections * wow * self.gamma * alpha
        self.connections = self.connections.clip(-limits, limits)
        self.gamma *= self.lr_decay

    # def visualize(self):
    #     for neuron in range(-100,-50):
    #         for steps in range(1, 2):
    #             maxer = (None, 0)
    #             matrixer = np.linalg.matrix_power(self.connections, steps)[:pixels**2 * channels, neuron]
    #             initimage = np.zeros((pixels, pixels, channels))
    #             for count in range(4):
    #                 for _ in range(500):
    #                     print("---")
    #                     image = (1 - 1 / np.power(2, count)) * initimage + 1 / np.power(2, count) * np.random.uniform(size=(pixels, pixels, channels))
    #                     firing = image.flatten() @ matrixer
    #                     if firing > maxer[1]:
    #                         maxer = (image, firing)
    #                 initimage = maxer[0]
    #             plt.imshow(((maxer[0] * 255).clip(0, 255)).astype(int))
    #             plt.title(str(steps)+","+str(neuron))
    #             plt.show()
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
        else:
            self.sight = self.screenshot()


count = 1000
dims = 3
percentile = 10
audiosize = 0
randomsize = 200
firing_history = 50
pixels = 14
channels = 1
audvis_count = pixels ** 2 * channels + audiosize
sensory_count = audvis_count + randomsize
output_count = 50
gamma = 0.02
threshold = 0.1
limits = 10
reward_amount = 2.0
decay = 1 - 1E-8
base_time = 1549200000
video_stream = False
exp_decay = np.power(gamma, -np.arange(firing_history - 1))[None, :, None]


def main():
    if video_stream:
        cam = cv2.VideoCapture(0)
        cam.set(3, 36)
        cam.set(4, 64)
    else:
        cam = None
    count = 0
    n = 100000
    executor = ThreadPoolExecutor(max_workers=3)
    george = Mind(executor, gamma)
    keyboard_press = Controller()

    def input():
        george.audiovision(cam)

    def output(keyboard):
        george.output(keyboard)

    def processing(count):
        george.fire()
        thought = george.firings[:, sensory_count:-output_count]
        nov = np.abs(thought[-1] - thought[-3]).mean() / thought.sum() * 2E5
        if nov > 0.001:
            executor.submit(george.learn, np.sqrt(nov))
        george.decay()
        if (count % n == n - 1):
            george.visualize()

    def show():
        if george.sight is not None:
            plt.imshow(george.sight)
            plt.show()

    while True:
        # show()
        executor.submit(input)
        count += 1
        executor.submit(processing, count)
        output(keyboard_press)
        time.sleep(0.01)
        # if keyboard.is_pressed("space"): #if key space is pressed.You can also use right,left,up,down and others like a,b,c,etc.
        #     print("Rewarded")
        #     george.reward()


main()
