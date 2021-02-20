import os
import argparse
import configparser
import logging
import tensorflow as tf
import threading
import multiprocessing
from simulator import Simulator
from model import Model
from trainer import (Counter, Trainer, Player,
                     check_dir, init_dir, init_log)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp_train = subparsers.add_parser('train', help='train agents')
    sp_train.add_argument('--gui', action='store_true', help="shows SUMO gui")
    sp_train.add_argument('--tfboard', action='store_true', help="enable logging to tensorboard")
    sp_train.add_argument('--port', type=int, required=False, default=0, help="port of sumo")
    sp_play=subparsers.add_parser('play', help="play the game")
    sp_play.add_argument('--port', type=int, required=False, default=0, help="port of sumo")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_simulator(config, port=0):
    env = Simulator(config, port=port)
    return env


def train(args):
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, 'out')
    dirs = init_dir(out_dir)
    init_log(dirs['log'])
    config_ini = os.path.join(cwd, 'config', 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_ini)

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, log_step)

    # number of worker agents
    no_of_workers = multiprocessing.cpu_count()/2  # num of agents 

    # init tf
    seed = config.getint('ENV_CONFIG', 'seed')
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # init simulator
    sim = init_simulator(config['ENV_CONFIG'], port=args.port)    
    # init agent    
    model = Model(sim.n_s_ls, sim.n_a_ls, sim.n_w_ls, sim.n_f_ls, total_step, sess,
                  config['MODEL_CONFIG'])

    with tf.device("/cpu:0"):
        sim = init_simulator(config['ENV_CONFIG'], port=args.port) # terminate inside
        global_ac = Model(sim.n_s_ls, sim.n_a_ls, sim.n_w_ls, sim.n_f_ls, total_step, sess,
                  config['MODEL_CONFIG'], name='global_ac')
        
        workers = []        
        # loop for each workers
        for i in range(no_of_workers):
            i_name = 'W_%i' % i
            port=args.port + i*2
            workers.append(Worker(sim.n_s_ls, sim.n_a_ls, sim.n_w_ls, sim.n_f_ls, total_step, global_ac, sess,
                  config['MODEL_CONFIG'], name= i_name, port=port))
    
    coord = tf.train.Coordinator()
    

    # model.load(out_dir + '/model/model')
    # print(sim.n_s_ls, sim.n_a_ls, sim.n_w_ls, sim.n_f_ls)
    summary_writer = None
    if args.tfboard:
        summary_writer = tf.summary.FileWriter(dirs['log'])
    
    # train from workers
    trainer = Trainer(sim, model, global_counter, summary_writer, output_path=dirs)
    trainer.run(args.gui)

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'])


def play_fn(agent_dir, seeds, port):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    
    # load config file for env
    cwd = os.getcwd()
    config_ini = os.path.join(cwd, 'config', 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_ini)

    # init simulator
    sim = init_simulator(config['ENV_CONFIG'], port=port)
    sim.episode_length_sec=1e6
    sim.init_test_seeds(seeds)
    
    # init tf
    tf.reset_default_graph()
    tf.set_random_seed(0)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model = Model(sim.n_s_ls, sim.n_a_ls, sim.n_w_ls, sim.n_f_ls, 0, sess, config['MODEL_CONFIG'])

    model.load(agent_dir + '/model/model')

    sim.agent = agent
    # play
    player = Player(sim, model)
    player.play()


def play(args):
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, 'out')

    # enforce the same evaluation seeds across agents
    threads = []

    thread = threading.Thread(target=play_fn,
                              args=(out_dir, [0], args.port))
    thread.start()
    threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        play(args)
