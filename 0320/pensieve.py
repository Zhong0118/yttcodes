import os
import time

import numpy as np
import tensorflow as tf
import pickle
from server import ServerBuffer
from model.sac_pic import SAC

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from simulator.base_abr import BaseAbr
from simulator.constants import (
    BUFFER_NORM_FACTOR,
    CRITIC_LR_RATE,
    DEFAULT_QUALITY,
    VIDEO_BIT_RATE,
    M_IN_K,
    VIDEO_CHUNK_LEN,
    MILLISECONDS_IN_SECOND,
    TOTAL_VIDEO_CHUNK,
    TRAIN_SEQ_LEN,
)
from simulator.env import Environment
from model import a3c

BITRATE_DIM = 6
RAND_RANGE = 1000


def entropy_weight_decay_func(epoch):
    # linear decay
    # return np.maximum(-0.05/(10**4) * epoch + 0.5, 0.1)
    return 0.5


def softmax(x):
    """ softmax function """
    x = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    return x


def learning_rate_decay_func(epoch):
    rate = 0.0001
    return rate


def noise_epi(epoch):
    return np.max([0.5 * np.e ** (-1 * epoch / 5000), 0.02])


def adjust_bw_co(co: list, online_num: int):
    for co_index in range(online_num):
        if co[co_index] < 0.001:
            max_index = int(np.argmax(co))
            co[max_index] -= 0.005
            co[co_index] += 0.005
    return co


def add_noise(action, num_agent):
    seed_index = np.random.randint(0, num_agent, 1)[0]
    action[0][seed_index] *= 5
    return softmax(action)


class Pensieve(BaseAbr):
    abr_name = "pensieve"

    def __init__(self, model_path: str = "", s_info: int = 6, s_len: int = 8,
                 a_dim: int = 6, plot_flag: bool = False, train_mode=False):
        """Penseive
        Input state matrix shape: [s_info, s_len]

        Args
            model_path: pretrained model_path.
            s_info: number of features in input state matrix.
            s_len: number of past chunks.
            a_dim: number of actions in action space.
        """
        self.s_info = s_info
        self.s_len = s_len
        self.a_dim = a_dim
        self.jump_action = False
        if self.s_info == 6 and self.s_len == 8 and self.a_dim == 6:
            print('use original pensieve')
        elif self.s_info == 6 and self.s_len == 6 and self.a_dim == 3:
            self.jump_action = True
            print('use jump action')
        else:
            raise NotImplementedError
        self.plot_flag = plot_flag
        self.train_mode = train_mode

    @staticmethod
    def get_next_bitrate(state, last_bit_rate, actor, evaluate):

        action_prob = actor.predict(state)
        action_cumsum = np.cumsum(action_prob)
        bit_rate = (
                action_cumsum
                > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
        ).argmax()
        if evaluate:
            bit_rate = np.argmax(action_prob)

        return bit_rate, action_prob

    def train(self, trace_scheduler, save_dir: str, total_epoch: int,
              video_size_file_dir: str, k: int, time_slot: int, device_type: list
              , pretrained_abr, pretrained_abr_dir: str
              , pretrained_net, pretrained_net_dir: str):
        assert self.train_mode

        vec_dim = 5
        num_agents = len(device_type)
        server = ServerBuffer(n_agents=num_agents, time_slot=time_slot, device_info=device_type, k=k)

        reward_lists = [[] for _ in range(num_agents)]
        actors = []
        critics = []

        qoe_matrix = [
            [0.11521495 * 4, 0.259233638 * 4, 0.539205966 * 4, 4, 4, 4],
            [0.216496929, 0.487118091, 1.013205629, 2.918378608, 4, 4],
            [0.118842995, 0.267396739, 0.556185218, 1.602003576, 3.604508046, 4],
            [0.066495485, 0.149614842, 0.311198872, 0.896359144, 2.016808073, 3.585436575],
        ]

        with tf.Session() as sess:
            for i in range(3):
                actor = a3c.ActorNetwork(sess,
                                         state_dim=[self.s_info, self.s_len],
                                         action_dim=self.a_dim,
                                         bitrate_dim=BITRATE_DIM,
                                         name='actor' + str(i))
                critic = a3c.CriticNetwork(sess,
                                           state_dim=[self.s_info, self.s_len],
                                           learning_rate=CRITIC_LR_RATE,
                                           bitrate_dim=BITRATE_DIM,
                                           name='critic' + str(i))
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(max_to_keep=None)  # save neural net parameters

                actors.append(actor)
                critics.append(critic)

            res_base_dir = './res/'

            exp_name = 'binets_slot=' + str(time_slot) + "_" + str(time.strftime("%m-%d %H-%M-%S", time.localtime()))

            base_bw = 0.15

            res_dir = res_base_dir + exp_name + '/'
            os.makedirs(res_dir, exist_ok=True)
            os.makedirs(res_dir + 'model_saved', exist_ok=True)

            sac = SAC(memory_capacity=4096, state_dim=k * vec_dim, action_dim=k,
                      model_dir=res_dir + 'model_saved/')
            if pretrained_net:
                sac.load_model(pretrained_net_dir)

            if pretrained_abr:
                model_file = tf.train.latest_checkpoint(pretrained_abr_dir)
                saver.restore(sess, model_file)
                print("Model restored.")

            for index in range(3, num_agents):
                actors.append(actors[device_type[index]])
                critics.append(critics[device_type[index]])

            os.makedirs(os.path.join(save_dir, "model_saved"), exist_ok=True)
            exp_queue = [[] for _ in range(num_agents)]
            net_envs = []

            for i in range(num_agents):
                net_env = Environment(trace_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                                      video_size_file_dir=video_size_file_dir,
                                      random_seed=1)
                net_envs.append(net_env)
            for env in net_envs:
                env.set_bw_co(base_bw)

            last_bit_rate = [DEFAULT_QUALITY for _ in range(num_agents)]
            bit_rate = [DEFAULT_QUALITY for _ in range(num_agents)]
            action_vec = np.zeros(self.a_dim)
            action_vec[bit_rate] = 1
            s_batch = [[np.zeros((self.s_info, self.s_len))] for _ in range(num_agents)]
            a_batch = [[action_vec] for _ in range(num_agents)]
            r_batch = [[] for _ in range(num_agents)]
            entropy_record = [[] for _ in range(num_agents)]

            update_tag = False
            epoch = 0

            bw_co_list = []
            bw_co_list.append([base_bw for _ in range(num_agents)])
            smooth_penalty = [0.2 for _ in range(num_agents)]
            rebuf_penalty = [10 for _ in range(num_agents)]

            server_state = []
            server_action = []
            wss_list = []
            co_bw = [1 / num_agents for _ in range(num_agents)]

            while epoch < total_epoch:

                if epoch % TOTAL_VIDEO_CHUNK == 0:
                    first_slot = True

                    for env in net_envs:
                        env.set_bw_co(base_bw)

                elif epoch % time_slot == 0:

                    server_state_, classf, server_reward, min_agent_index, wss = server.get_previous_state()

                    if first_slot:
                        first_slot = False
                    else:
                        sac.M.store_transition(server_state, server_action, np.array([server_reward]), server_state_)
                        server.r_list.append(server_reward)
                    server_state = server_state_

                    print(str(server_reward) + " " + str(min_agent_index) + " " + str(co_bw))

                    server_action = sac.actor.choose_action(server_state)

                    co_t = list(softmax(server_action))
                    co_bw = [0.0 for _ in range(num_agents)]
                    for i in range(len(co_t)):
                        for elem in classf[i]:
                            co_bw[elem] = co_t[i]

                    sum_t = np.sum(co_bw)
                    for i in range(len(co_bw)):
                        co_bw[i] = co_bw[i] / sum_t

                    co_bw = adjust_bw_co(co_bw, len(device_type))

                    bw_co_list.append(co_bw)
                    if wss >= 0:
                        wss_list.append(wss)

                    for env, bw in zip(net_envs, co_bw):
                        env.set_bw_co(bw * base_bw * num_agents)
                if epoch % time_slot == 0 and not pretrained_net:
                    sac.update(epoch=int(epoch / time_slot))

                if epoch % 20000 == 0:
                    rew_server = res_dir + "server_reward.pkl"
                    with open(rew_server, 'wb') as fp:
                        pickle.dump(server.r_list, fp)
                    bw_co_file = res_dir + "bw_co.pkl"
                    with open(bw_co_file, 'wb') as fp:
                        pickle.dump(bw_co_list, fp)

                    wss_value = res_dir + "wss_value.pkl"
                    with open(wss_value, 'wb') as fp:
                        pickle.dump(wss_list, fp)
                    sac.save_model()

                for i in range(num_agents):
                    delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain = \
                        net_envs[i].get_video_chunk(bit_rate[i])

                    reward = qoe_matrix[device_type[i]][bit_rate[i]] - rebuf_penalty[i] * rebuf - \
                                 smooth_penalty[i] * np.abs(bit_rate[i] - last_bit_rate[i])

                    r_batch[i].append(reward)
                    last_bit_rate[i] = bit_rate[i]

                    state = np.array(s_batch[i][-1], copy=True)
                    state = np.roll(state, -1, axis=1)

                    state[0, -1] = VIDEO_BIT_RATE[bit_rate[i]] / float(np.max(VIDEO_BIT_RATE))  # last quality
                    state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                    state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                    state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                    state[4, :BITRATE_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                    state[5, -1] = np.minimum(video_chunk_remain, TOTAL_VIDEO_CHUNK) / float(TOTAL_VIDEO_CHUNK)

                    bit_rate[i], action_prob = Pensieve.get_next_bitrate(
                        np.reshape(state, (1, self.s_info, self.s_len)),
                        bit_rate[i],
                        actors[i],
                        pretrained_net)

                    entropy_record[i].append(a3c.compute_entropy(action_prob[0]))

                    server.send_msg(qoe=reward, bitrate=bit_rate[i], delay=delay, buff_size=buffer_size,
                                    throughput=float(video_chunk_size) / float(delay), index=i)

                    if pretrained_net and epoch > 100000:
                        exit()

                    if len(r_batch[i]) >= TRAIN_SEQ_LEN or end_of_video:
                        exp_queue[i].append([s_batch[i][1:],  # ignore the first chuck
                                             a_batch[i][1:],  # since we don't have the
                                             r_batch[i][1:],  # control over it
                                             end_of_video,
                                             {'entropy': entropy_record[i]}])

                        s_batch[i] = [np.zeros((self.s_info, self.s_len))]
                        a_batch[i] = [action_vec]
                        r_batch[i] = []
                        entropy_record[i] = []
                        update_tag = True
                        continue

                    s_batch[i].append(state)
                    action_vec = np.zeros(self.a_dim)
                    action_vec[bit_rate[i]] = 1
                    a_batch[i].append(action_vec)

                epoch += 1

                total_reward = 0.0
                total_entropy = 0.0

                entropy_weight = entropy_weight_decay_func(epoch)
                current_learning_rate = learning_rate_decay_func(epoch)

                if update_tag:
                    for i in range(num_agents):
                        if len(exp_queue[i]) != 0:
                            s_batch_t, a_batch_t, r_batch_t, terminal_t, info_t = exp_queue[i][-1]
                            del exp_queue[i][-1]

                            if not pretrained_net:
                                actor_gradient, critic_gradient, td_batch = a3c.compute_gradients(
                                    s_batch=np.stack(s_batch_t, axis=0),
                                    a_batch=np.vstack(a_batch_t),
                                    r_batch=np.vstack(r_batch_t),
                                    terminal=terminal_t, actor=actors[i],
                                    critic=critics[i],
                                    entropy_weight=entropy_weight)
                                actors[i].apply_gradients(actor_gradient, current_learning_rate)
                                critics[i].apply_gradients(critic_gradient)

                            total_reward += np.sum(r_batch_t)

                            total_entropy += np.sum(info_t['entropy'])
                            avg_reward = np.sum(r_batch_t) / len(r_batch_t)
                            reward_lists[i].append(avg_reward)

                    update_tag = False

                if epoch % 50000 == 0:
                    for a_index in range(num_agents):
                        rew_file_name = res_dir + "reward" + str(a_index) + ".pkl"
                        with open(rew_file_name, 'wb') as fp:
                            pickle.dump(reward_lists[a_index], fp)


def calculate_from_selection(selected, last_bit_rate):
    # naive step implementation
    # action=0, bitrate-1; action=1, bitrate stay; action=2, bitrate+1
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    bit_rate = max(0, bit_rate)
    bit_rate = min(5, bit_rate)

    return bit_rate
