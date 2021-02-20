import logging
import numpy as np
import subprocess
from sumolib import checkBinary
import time
import random
import traci
from tls_data import PHASESETS, PHASEMAPPINGS, NODES

DEFAULT_PORT = 8000

STATE_NAMES = ['wave', 'wait']


class PhaseSet:
    def __init__(self, phase_set):
        self.num_phase = len(phase_set)
        self.num_lane = len(phase_set[0])
        self.phases = phase_set

class PhaseSetInfo:
    def __init__(self):
        self.phase_sets = {}
        for phase_set_key, phase_set in PHASESETS.items():
            self.phase_sets[phase_set_key] = PhaseSet(phase_set)

    def get_phase(self, phase_set_key, action):
        # phase_type is either green or yellow
        return self.phase_sets[phase_set_key].phases[int(action)]

    def get_phase_num(self, phase_set_key):
        return self.phase_sets[phase_set_key].num_phase

    # def get_red_lanes(self, phase_set_id, action):
    #     # the lane number is link number
    #     return self.phase_sets[phase_set_id].red_lanes[int(action)]

class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control # disabled
        self.lanes_in = []
        self.ilds_in = [] # for state
        self.fingerprint = [] # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0 
        self.num_fingerprint = 0
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        self.phase_set_key = -1
        self.phase_mapping_key = -1
        self.n_a = 0
        self.prev_action = -1
        self.last_actions = [0,0]
        self.distances = dict()
        self.false_transition = False

        graph = dict([(key, val[2]) for key, val in NODES.items()])
        for dest_name in graph.keys():
            self.distances[dest_name] = self._find_shortest_distance(graph, name, dest_name)

    @staticmethod
    # Breadth first search to find the shortest path between two nodes of a graph
    def _find_shortest_distance(graph, start, goal):
        visited = []
        queue = [[start]]
        if start == goal:
            return 0

        while queue:
            path = queue.pop(0)
            node = path[-1]

            if node not in visited:
                neighbours = graph[node]

                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)

                    if neighbour == goal:
                        return len(new_path) - 1
                visited.append(node)
        # return 100 if there is no connection
        return 1e2


class Simulator:
    def __init__(self, config, port=0):
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self.init_test_seeds(test_seeds)
        self._init_sim(self.seed)
        self._init_nodes()
        self.terminate()

    def _get_node_phase_set_key(self, node_name):
        return self.phase_set_key_info[node_name]

    def _get_node_phase_mapping_key(self, node_name):
        return NODES[node_name][1]

    def _get_node_state_num(self, node):
        # wait / wave states for each lane
        return len(node.ilds_in)

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def _init_nodes(self):
        nodes = {}
        for node_name in self.sim.trafficlight.getIDList():
            if node_name in self.neighbor_map: # 
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found!' % node_name)
                neighbor = []
            nodes[node_name] = Node(node_name,
                                    neighbor=neighbor,
                                    control=True)
            # controlled lanes: l:j,i_k
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            lanes_in = tuple([t for t in list(lanes_in) if not t.startswith(':')]) # remove crosswalk
            nodes[node_name].lanes_in = lanes_in
            ilds_in = []
            for lane_name in lanes_in:
                ild_name = lane_name
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)
            # nodes[node_name].edges_in = edges_in
            nodes[node_name].ilds_in = ilds_in
        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            s += '\tilds_in: %r\n' % node.ilds_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_a_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_set_key = self._get_node_phase_set_key(node_name)
            phase_mapping_key = self._get_node_phase_mapping_key(node_name)
            node.phase_set_key = phase_set_key
            node.phase_mapping_key = phase_mapping_key
            node.n_a = self.phase_set_info.get_phase_num(phase_set_key) # 6 actions
            self.n_a_ls.append(node.n_a)
        # for global coop level
        self.n_a = np.prod(np.array(self.n_a_ls))

    def _init_map(self):
        self.neighbor_map = dict([(key, val[2]) for key, val in NODES.items()])
        self.phase_set_info = PhaseSetInfo()
        self.phase_set_key_info = dict([(key, val[0]) for key, val in NODES.items()])
        self.state_names = STATE_NAMES

    def _init_policy(self): # uniform policy
        policy = []
        for node_name in self.node_names:
            phase_num = self.nodes[node_name].n_a
            p = 1. / phase_num
            policy.append(np.array([p] * phase_num))
        return policy

    def _init_sim(self, seed, gui=False):
        sumocfg_file = "./config/pingshan.sumocfg"
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '600'] # teleport for safety
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        if gui:
            command += ['-S', 'True']
            command += ['-Q', 'True']
        if self.train_mode:
            command += ['--scale', '1.3']
        # collect trip info if necessary
        subprocess.Popen(command)
        # wait 2s to establish the traci server
        time.sleep(2)
        self.sim = traci.connect(port=self.port)
        if gui:
            self.sim.gui.setSchema("View #0", "real world")

    def _init_sim_traffic(self):
        return

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        self.n_w_ls = []
        self.n_f_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            num_wave = node.num_state
            num_fingerprint = 0
            for nnode_name in node.neighbor:
                num_wave += self.nodes[nnode_name].num_state
                num_fingerprint += self.nodes[nnode_name].num_fingerprint
            # wait time, halt time, jam vehicle, mean speed
            num_wait = 0 if 'wait' not in self.state_names else node.num_state*5
            self.n_s_ls.append(num_wave + num_wait + num_fingerprint)
            self.n_f_ls.append(num_fingerprint)
            self.n_w_ls.append(num_wait)
        self.n_s = np.sum(np.array(self.n_s_ls))


    @staticmethod
    def _norm_clip_state(x, norm, clip=1):
        x = x / norm
        return np.clip(x, -clip, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0
            node.last_actions = [0,0]
            node.false_transition = False
            node.num_fingerprint = node.n_a # use current mapping as fingerprint
            node.num_state = self._get_node_state_num(node)
    
    ## step_function
    # use in step
    def _set_phase(self, action, phase_type, phase_duration):
        for node_name, a in zip(self.node_names, list(action)):
            phase = self._get_node_phase(a, node_name, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)
    
    # use in step
    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]
        cur_phase = self.phase_set_info.get_phase(node.phase_set_key, action)
        if phase_type == 'green':
            return cur_phase
        prev_action = node.prev_action
        
        node.prev_action = action
        #fingerprint must be updated before
        node.false_transition = not node.fingerprint[action]
        if action != node.last_actions[-1]:
            node.last_actions.append(action)
            node.last_actions.pop(0)
                   
        if (prev_action < 0) or (action == prev_action):
            return cur_phase
        prev_phase = self.phase_set_info.get_phase(node.phase_set_key, prev_action)
        switch_reds = []
        switch_greens = []
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):
                switch_greens.append(i)
        if not len(switch_reds):
            return cur_phase
        yellow_phase = list(cur_phase)
        for i in switch_reds:
            yellow_phase[i] = 'y'
        for i in switch_greens:
            yellow_phase[i] = 'r'
        return ''.join(yellow_phase)
    
    # use in step
    def _simulate(self, num_step):
        for _ in range(num_step):
            self.sim.simulationStep()
            self.cur_sec += 1
            
    # use in step
    def _measure_reward_step(self):
        rewards = []
        for node_name in self.node_names:
            if self.nodes[node_name].false_transition:
                reward = -300
            else:
                queues = []
                waits = []
                for ild in self.nodes[node_name].ilds_in:
                    if self.obj in ['queue', 'hybrid']:
                        cur_queue = self.sim.lanearea.getLastStepHaltingNumber(ild)
                        if self.sim.lanearea.getJamLengthMeters(ild) > 80:
                            cur_queue += 100
                        queues.append(cur_queue)
                    if self.obj in ['wait', 'hybrid']:
                        max_pos = 0
                        car_wait = 0
                        cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                        for vid in cur_cars:
                            car_pos = self.sim.vehicle.getLanePosition(vid)
                            if car_pos > max_pos:
                                max_pos = car_pos
                                car_wait = self.sim.vehicle.getWaitingTime(vid)
                        waits.append(car_wait)
    
                queue = np.sum(np.array(queues)) if len(queues) else 0
                wait = np.sum(np.array(waits)) if len(waits) else 0
                if self.obj == 'queue':
                    reward = - queue
                elif self.obj == 'wait':
                    reward = - wait
                else:
                    reward = - queue - self.coef_wait * wait
            rewards.append(reward)
        return np.array(rewards)
    
    # use in step
    def _get_state(self):
        # the state ordering as wave, wait, fp
        state = []
        # measure the most recent state
        self._measure_state_step() #update state for nodes

        # get the appropriate state vectors
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # wave is required in state
            cur_state = [node.wave_state]
            # include wave states of neighbors
            for nnode_name in node.neighbor:
                # discount the neigboring states
                cur_state.append(self.nodes[nnode_name].wave_state * self.coop_gamma)
            # include wait state
            cur_state.append(node.wait_state)
            # include fingerprints of neighbors
            for nnode_name in node.neighbor:
                cur_state.append(self.nodes[nnode_name].fingerprint)
            state.append(np.concatenate(cur_state))

        return state
    
    # use in step
    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:
                if state_name == 'wave':
                    cur_state = []
                    for ild in node.ilds_in:
                        cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)
                        cur_state.append(cur_wave)
                    cur_state = np.array(cur_state)
                else:
                    cur_occ = []
                    cur_halt = []
                    cur_speed = []
                    cur_jam_v = []
                    cur_jam_m = []
                    # cur_jam_p = []
                    for ild in node.ilds_in:

                        cur_occ.append(self._norm_clip_state(self.sim.lanearea.getLastStepOccupancy(ild),norm=1,clip=1))
                        cur_halt.append(self._norm_clip_state(self.sim.lanearea.getLastStepHaltingNumber(ild),norm=1,clip=100))
                        cur_speed.append(self._norm_clip_state(self.sim.lanearea.getLastStepMeanSpeed(ild),norm=1,clip=30))
                        cur_jam_v.append(self._norm_clip_state(self.sim.lanearea.getJamLengthVehicle(ild),norm=1,clip=200))
                        cur_jam_m.append(self._norm_clip_state(self.sim.lanearea.getJamLengthMeters(ild),norm=1,clip=200))
                        # cur_jam_p.append(cur_occ[-1] > 60)
                    cur_state = cur_occ + cur_halt + cur_speed + cur_jam_v + cur_jam_m 
                    cur_state = np.array(cur_state)
                
                if state_name == 'wave':
                    node.wave_state = cur_state
                else:
                    node.wait_state = cur_state
                    
    # use in run func (main)
    def reset(self, gui=False, test_ind=0):
        # have to terminate previous sim before calling reset
        self._reset_state()
        if self.train_mode:
            # seed = self.seed
            seed = random.randint(0, 1e5)
        else:
            seed = self.test_seeds[test_ind]
        # self._init_sim(gui=True)
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self.cur_episode += 1
        # initialize fingerprint
        self._init_policy()
        self.update_fingerprint()
        self._init_sim_traffic()
        return self._get_state()

    # use in run func (main)
    def terminate(self):
        self.sim.close()

    def step(self, action):
        self._set_phase(action, 'yellow', self.yellow_interval_sec)
        self._simulate(self.yellow_interval_sec)
        rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
        self._set_phase(action, 'green', rest_interval_sec)
        self._simulate(rest_interval_sec)
        state = self._get_state()
        reward = self._measure_reward_step()
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True
        global_reward = np.sum(reward) # for fair comparison

        # use local rewards in test
        if not self.train_mode:
            return state, reward, done, global_reward
        # discounted global reward
        new_reward = []
        for node_name, r in zip(self.node_names, reward):
            cur_reward = r
            for i, nnode in enumerate(self.node_names):
                if nnode == node_name:
                    continue
                else:
                    distance = self.nodes[node_name].distances[nnode]
                    cur_reward += (self.coop_gamma ** distance) * reward[i]
            new_reward.append(cur_reward)
        reward = np.array(new_reward)
        return state, reward, done, global_reward

    def update_fingerprint(self):
        # for node_name, pi in zip(self.node_names, policy):
        #     self.nodes[node_name].fingerprint = np.array(pi)[:-1]
        for node_name in self.node_names:
            mappings = PHASEMAPPINGS[self.nodes[node_name].phase_mapping_key]
            last_actions = self.nodes[node_name].last_actions
            mapping = mappings[last_actions[-1]][last_actions[-2]]
            self.nodes[node_name].fingerprint =  np.array([int(s) for s in mapping])           
            
            
            
            
            
            
            
            
            

