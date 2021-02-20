import itertools
import logging
import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def init_dir(base_dir, pathes=['log', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


class Counter:
    def __init__(self, total_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.batch_size = self.model.batch_size
        self.plans = [[] for _ in range(len(self.env.nodes))]
        self.summary_writer = summary_writer
        assert self.env.T % self.batch_size == 0
        self.data = []
        self.output_path = output_path
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        if self.summary_writer:
            self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        global_rewards = []
        for _ in range(self.batch_size):
            policy, value = self.model.forward(ob, done)
            # need to update fingerprint before calling step
            self.env.update_fingerprint()
            global_step = self.global_counter.next()

            action = []
            for i, pi in enumerate(policy):
                
                action.append(np.random.choice(np.arange(len(pi)), p=pi)) #sample using pi distribution

            next_ob, agent_rewards, done, global_reward = self.env.step(action)
            global_rewards.append(global_reward)

            self.cur_step += 1
            self.model.add_transition(ob, action, agent_rewards, value, done)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(agent_rewards), done))
            if done:
                break
            ob = next_ob
        if done:
            R = [0] * self.model.n_agent
        else:
            R = self.model.forward(ob, False, 'v')
        return ob, done, R, global_rewards

    def run(self, gui=False):
        while not self.global_counter.should_stop(): # total steps
            # train
            self.env.train_mode = True
            ob = self.env.reset(gui)
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            rewards = []
            while True: # each episode
                ob, done, R, cur_rewards = self.explore(ob, done)
                rewards += cur_rewards
                global_step = self.global_counter.cur_step
                self.model.backward(R, self.summary_writer, global_step)
                # termination
                if done:
                    self.env.terminate()
                    break
            rewards = np.array(rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            if self.summary_writer:
                self.summary_writer.flush()
                
            if global_step%1000 < 1e-2:
                self.model.save(self.output_path['model'])
                
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path['log'] + 'train_reward.csv')


class Player:
    def __init__(self, sim, model):
        self.env = sim
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.plans = [[] for _ in range(len(self.env.nodes))]

    def play(self):
        self.env.cur_episode = 0
        time.sleep(1)
        state = self.env.reset(gui=True, test_ind=0)
        done = True
        self.model.reset()
        while True:
            policy = self.model.forward(state, done, 'p')
            self.env.update_fingerprint()
            action = []
            for i, pi in enumerate(policy):
                # action.append(np.random.choice(np.arange(len(pi)), p=pi))
                action.append(np.argmax(np.array(pi)))
                # a = self.planned_action(i, pi)
                # action.append(a)
            next_ob, reward, done, global_reward = self.env.step(action)
            if done:
                break
            state = next_ob
        self.env.terminate()
        time.sleep(2)

