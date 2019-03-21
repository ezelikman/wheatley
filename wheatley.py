import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

## remove below!!
# dims = 3 # Number of dimensions of the neuron space
# percentile = 100  # What portion of neurons (by distance) should a neuron connect to
#
# total_n = 50 # Number of total neurons
# audio_n = 0 # Number of audio neurons
# reward_n = 0  # Number of reward-perceiving neurons
# output_n = 25  # Number of output neurons (averaged to determine output)
# random_n, random_p = 2, 1  # Number and likelihood of randomly firing neurons
# repeats = 1  # Number of times to repeat input
#
# firing_history = 2000 # How many time-steps of firing to remember, used to threshold firing
# # exp_decay = np.power(init_gamma, -np.arange(firing_history - 1))[None, :, None]
# long_plasticity = False
# limits = 2 # Maximum connection strength between neurons (+-)

class Mind:
    def __init__(self, threader, mode="gym", long_plasticity=False, dims=3, percentile=100, base_n=4,
                    total_n=50, audio_n=0, reward_n=0, output_n=25, random_n=2, random_p=1, repeats=1,
                    firing_history=2000, limits=2, intermediate_loops=False, recurrent_out_loops=True,
                    gamma=0.002, video_stream=False):
        '''
        Params
        threader: threader object
        mode: idk
        long_plasticity: hmm
        dims: number of dimensions of the neuron space
        percentile: what portion of neurons (by distance) should a neuron connect to
        base_n: number of base neurons
        total_n: number of total neurons
        audio_n: number of audio neurons
        reward_n: number of reward-perceiving neurons
        output_n: number of output neurons (averaged to determine output)
        random_n, random_p: number and likelihood of randomly firing neurons
        repeats: number of times to repeat input
        firing_history: how many time-steps of firing to remember, used to threshold firing
        limits: maximum connection strength between neurons (+-)
        intermediate_loops: whether to enable/disable intermediate loops (bool)
        recurrent_out_loops: whether to enable/disable output loops back into the network (bool)
        gamma: decay factor
        video_stream: reading from visual input (bool)

        '''

        self.video_stream = video_stream
        self.mode = mode
        self.pixels, self.channels = (10, 1) if video_stream else (None, None)  # Dimensions of visual input
        self.limits = limits
        self.repeats = repeats
        self.base_n = base_n
        self.total_n = total_n
        self.audio_n = audio_n
        self.reward_n = reward_n
        self.output_n = output_n
        self.random_n, self.random_p = random_n, random_p
        self.video_n = self.base_n * repeats if self.pixels is None else self.pixels ** 2 * self.channels
        self.audvis_n = self.video_n + self.audio_n # Includes the perceptive neurons
        self.sensory_n = self.audvis_n + self.random_n + self.reward_n # Number of inputs in total

        self.neurons = np.random.uniform(size=(self.total_n, dims))  # Physical position in space
        self.neurons[-1][-self.output_n:] = 0.9
        if self.pixels is not None:
            visual_map = np.mgrid[0.3:0.7:self.pixels * 1j, 0.3:0.7:self.pixels * 1j, 0.2:0.4:self.channels * 1j]
        else:
            visual_map = np.mgrid[0.3:0.7:self.base_n * 1j, 0.3:0.7:repeats * 1j, 0.29:0.31:1j]
        visual_map = visual_map.reshape(dims, -1).T
        self.mode = mode
        self.xor = np.zeros(self.video_n)
        self.neurons[:self.video_n] = visual_map
        self.dists = np.sqrt(np.mean(np.square(self.neurons[None, :, :] - self.neurons[:, None, :]), axis=-1))
        self.firings = np.random.binomial(size=(firing_history, self.total_n), n=1, p=0.5).astype(float)
        self.connections = (
                (self.dists > 0) *
                (self.dists <= np.percentile(self.dists, percentile)) *
                np.random.uniform(size=self.dists.shape, low=-1, high=1)
        )

        # Disable connections coming into the sensory neurons
        self.connections[:, :self.sensory_n] = 0
        # # Disable direct connections from the inputs to the outputs
        self.connections[:self.sensory_n, -self.output_n:] = 0
        # Disable intermediate loops:
        if intermediate_loops == False:
            self.connections[self.sensory_n:, self.sensory_n:-self.output_n] = 0
        # # Disable connections coming out of the outputs
        if recurrent_out_loops == False:
            self.connections[-self.output_n:, :] = 0

        self.plastic = np.ones_like(self.connections)
        self.plastic[self.connections == 0] = 0
        self.init_gamma = gamma
        self.gamma = gamma
        self.threader = threader
        #self.lr_decay = 1 - self.gamma / 100
        self.con_decay = 1 - self.gamma / 100 # How much connections decay every time-step
        self.lr_decay = 0.99
        self.acc_decay = 1 - self.gamma
        self.screen_cur = None
        self.screen_prev = None
        self.iter_num = 0
        self.reward = 0

        # only for debugging!!
        self.prev_connections = None
        self.prev_connected = False

        # Should plasticity change to create long-term memory connections (Bool)
        self.long_plasticity = long_plasticity
        # How much has a neuron been updated over some number of time-steps
        self.accumulation = np.ones_like(self.connections)
        self.accumulation /= 1 - self.acc_decay
        self.up = np.zeros_like(self.firings[0, self.sensory_n:])
        self.down = np.zeros_like(self.firings[0, self.sensory_n:])
        self.sight = None
        self.sound = None

        # Just to store performance, not used by agent
        self.performance = np.asarray([])

    def output(self, keyboard_to_press):
        # if self.firings[-1][-self.output_n:].mean() > self.firings[:, -self.output_n:].mean():
        # if np.random.binomial(1, self.firings[-1][-self.output_n:].mean()) > 0.5:
        if self.firings[-1][-self.output_n:].mean() >= self.firings[:, -self.output_n:].mean():
            keyboard_to_press.press(" ")
        else:
            keyboard_to_press.release(" ")

    def fire(self):
        if self.sight is not None:
            visual_input = self.sight.flatten()
            firings_next = ((self.firings[-1] @ self.connections) > self.firings.mean(0)).astype(float)
            # firings_next = ((self.firings[-1] @ self.connections) > threshold).astype(float)
            # firings_next[:len(visual)] = visual > visual.mean()
            if self.mode is "dino" or self.mode is "cam":
                firings_next[:len(visual_input)] = visual_input / visual_input.max()
            else:
                firings_next[:len(visual_input)] = visual_input
            if self.reward_n > 0:
                firings_next[self.audvis_n:self.audvis_n + self.reward_n] = self.reward
            if self.audio_n > 0:
                firings_next[len(visual_input):self.audvis_n] = self.sound > self.sound.mean()
            if self.random_n > 0:
                firings_next[self.audvis_n:self.sensory_n] = np.random.binomial(size=(self.random_n,), n=1, p=self.random_p)
            # Update history
            self.firings[:-1] = self.firings[1:]
            self.firings[-1] = firings_next

    def plot(self):
        plt.close()
        #print(self.connections)
        if self.prev_connected == False:
            self.prev_connected = True
        else:
            print(self.prev_connections)
            print(self.connections)
            vals = np.nan_to_num(self.prev_connections / self.connections)
            vals = vals.flatten()
            vals = vals[np.nonzero(vals)]
            #print(vals)
            #vals[vals == np.nan] = 0
            # print(vals)
            print(np.std(vals))
        self.prev_connections = self.connections.copy()

        # print(np.linalg.norm(self.connections))
        plt.imshow(self.connections[:,:]/self.limits + 0.5)
        # plt.imshow(np.concatenate((self.connections[:,:,None]/self.limits + 0.5, self.connections[:,:,None], self.plastic[:,:,None]), axis=2))

        plt.savefig(self.mode + "4_" + str(self.iter_num) + ".png")

        plt.close()

        if self.pixels is not None:
            plt.imshow(
                ((self.connections * self.firings.mean(0)).sum(1) /
                 self.firings.sum())[:self.video_n].reshape(self.pixels, self.pixels))
            plt.savefig("dinoboi_" + str(self.iter_num) + ".png")
            plt.show()
            plt.close()

    # Weaken old connections over time
    def decay(self):
        if self.iter_num % 5000 == 0:
            self.plot()
        self.iter_num += 1
        self.connections *= self.con_decay

    def stdp(self, a, b):
        # print("STDP", a, b)
        a = np.asarray(a)[:, None]
        b = np.asarray(b)[None, :]
        c = b - (1 - a)
        d = (a * b).round().astype(int)
        e = np.multiply(c, (d | d.T).astype(float))
        return e

    def reinforce(self, alpha, hist=4):
        a = np.abs(self.connections).mean()
        for i in range(hist):
            if alpha < 0:
                self.connections[:, self.firings[-i-1].astype(bool)] /= 1 + self.gamma * np.abs(alpha) / hist
            else:
                self.connections[:, self.firings[-i-1].astype(bool)] *= 1 + self.gamma * np.abs(alpha) / hist
        self.connections *= a / np.abs(self.connections).mean()
        self.connections = self.connections.clip(-self.limits, self.limits)

    def learn(self, alpha=1.0):
        wow = np.multiply(np.multiply(self.stdp(self.firings[-2], self.firings[-1]), self.plastic), self.connections)

        if self.long_plasticity:
            # print(self.accumulation)
            # self.accumulation += np.abs(wow * self.gamma * alpha)
            # self.accumulation *= self.acc_decay
            # print(self.accumulation)
            # synapse_strength = np.abs(self.connections)
            # updates = (self.accumulation <= 0.01 / (1 - self.acc_decay)) * \
            #           (synapse_strength > 2 * (synapse_strength * self.plastic).sum() / self.plastic.sum())
            # print(updates)
            # self.plastic[self.sensory_n:, self.sensory_n:-self.output_n][updates[self.sensory_n:, self.sensory_n:-self.output_n].astype(bool)] = 0
            # print(self.plastic)
            # input()
            self.plastic[:, self.firings[-1].astype(bool)] *= 1 - self.gamma

        self.connections += self.connections * wow * self.gamma * alpha # * wow.sum() / 10
        self.connections = self.connections.clip(-self.limits, self.limits)
        # print(self.connections)
        self.gamma *= self.lr_decay

    def visualize(self):
        for neuron in range(-100,-90):
            try:
                sumer_max = np.zeros((self.pixels, self.pixels, self.channels))
                sumer_min = np.zeros((self.pixels, self.pixels, self.channels))
                sumer_diff = np.zeros((self.pixels, self.pixels, self.channels))
                for _ in range(200):
                    maxer = (-1, None)
                    miner = (10000, None)
                    initimage_max = np.zeros((self.pixels, self.pixels, self.channels))
                    initimage_min = np.zeros((self.pixels, self.pixels, self.channels))
                    for count in range(6):
                        for _ in range(100):
                            image_max = (1 - 1 / np.power(1.2, self.total_n)) * initimage_max + \
                                        1 / np.power(1.2, self.total_n) * np.random.uniform(size=(self.pixels, self.pixels, self.channels))
                            image_min = (1 - 1 / np.power(1.2, self.total_n)) * initimage_min + \
                                        1 / np.power(1.2, self.total_n) * np.random.uniform(size=(self.pixels, self.pixels, self.channels))
                            fire_max = np.zeros_like(self.firings[-1])
                            fire_min = np.zeros_like(self.firings[-1])
                            for c in range(3):
                                # fire[:self.pixels ** 2 * self.channels] = (image > image.mean()).flatten()
                                fire_max[:self.pixels ** 2 * self.channels] = image_max.flatten()
                                fire_min[:self.pixels ** 2 * self.channels] = image_min.flatten()
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
                    # plt.plot(self.neurons[neuron][0] * (self.pixels - 1), self.neurons[neuron][1] * (self.pixels - 1), 'bo')
                    # plt.show()
                    # plt.imshow(((miner[1] * 255).clip(0, 255)).astype(int))
                    # plt.plot(self.neurons[neuron][0] * (self.pixels - 1), self.neurons[neuron][1] * (self.pixels - 1), 'bo')
                    # plt.show()
                    diff = maxer[1] - miner[1]
                    sumer_diff += diff
                    sumer_max += maxer[1]
                    sumer_min += miner[1]
                    # plt.imshow((diff - diff.min())/(diff.max() - diff.min()))
                    # plt.plot(self.neurons[neuron][0] * (self.pixels - 1), self.neurons[neuron][1] * (self.pixels - 1), 'bo')
                    # plt.title(str(neuron))
                    # plt.show()
                    print("----")
                plt.imshow((sumer_max - sumer_max.min()) / (sumer_max.max() - sumer_max.min()))
                plt.plot(self.neurons[neuron][0] * self.pixels - 0.5, self.neurons[neuron][1] * self.pixels - 0.5, 'bo')
                plt.title("Max" + str(neuron))
                plt.show()

                plt.imshow((sumer_min - sumer_min.min()) / (sumer_min.max() - sumer_min.min()))
                plt.plot(self.neurons[neuron][0] * self.pixels - 0.5, self.neurons[neuron][1] * self.pixels - 0.5, 'bo')
                plt.title("Min" + str(neuron))
                plt.show()

                plt.imshow((sumer_diff - sumer_diff.min()) / (sumer_diff.max() - sumer_diff.min()))
                plt.plot(self.neurons[neuron][0] * self.pixels - 0.5, self.neurons[neuron][1] * self.pixels - 0.5, 'bo')
                plt.title("Diff" + str(neuron))
                plt.show()
            except:
                pass

    def screenshot(self):
        sct = mss()
        im = sct.grab({"top":100 ,"left":0, "width":500, "height":230})
        rgb = Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")
        full = np.array(rgb)
        visual = rgb.resize(size=(self.pixels, self.pixels), resample=Image.BICUBIC)
        visual = np.array(visual)
        grayscale = (visual.mean(-1)[:,:,None] - visual.min()) / (visual.max() - visual.min())
        # plt.imshow(visual)
        # plt.show()
        return grayscale, full

    def audiovision(self, cam, mic=None):
        if self.audio_n > 0:
            import pyaudio
            p = pyaudio.PyAudio()
            mic = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=self.audio_n)
            data = mic.read(self.audio_n)
            self.sound = np.fromstring(data, dtype=np.int16)
        else:
            self.sound = None

        if cam is not None:
            ret, frame = cam.read()
            base = cv2.resize(frame, dsize=(self.pixels, self.pixels), interpolation=cv2.INTER_AREA)
            self.full, self.sight = base.mean(-1)[:, :, None]
        elif self.mode == "dino":
            self.sight, self.full_image = self.screenshot()
            self.screen_prev = self.screen_cur
            self.screen_cur = self.full_image.copy()
        elif self.mode == "xor":
            self.sight = self.xor
