#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:25:33 2021

@author: changxi
"""

class Worker(object):
    def __init__(self, name, globalAC, sess):
        # intialize environment for each worker
        self.env = gym.make('MountainCarContinuous-v0').unwrapped
        self.name = name
        
        # create ActorCritic agent for each worker
        self.AC = ActorCritic(name, sess, globalAC)
        self.sess=sess
        
    def work(self):
        global global_rewards, global_episodes
        total_step = 1
 
        # store state, action, reward
        buffer_s, buffer_a, buffer_r = [], [], []
        
        # loop if the coordinator is active and global episode is less than the maximum episode
        while not coord.should_stop() and global_episodes < no_of_episodes:
            
            # initialize the environment by resetting
            s = self.env.reset()
            
            # store the episodic reward
            ep_r = 0
            for ep_t in range(no_of_ep_steps):
    
                # Render the environment for only worker 1
                if self.name == 'W_0' and render:
                    self.env.render()
                    
                # choose the action based on the policy
                a = self.AC.choose_action(s)

                # perform the action (a), recieve reward (r) and move to the next state (s_)
                s_, r, done, info = self.env.step(a)
             
                # set done as true if we reached maximum step per episode
                done = True if ep_t == no_of_ep_steps - 1 else False
                
                ep_r += r
                
                # store the state, action and rewards in the buffer
                buffer_s.append(s)
                buffer_a.append(a)
                
                # normalize the reward
                buffer_r.append((r+8)/8)
    
    
                # we Update the global network after particular time step
                if total_step % update_global == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    
                    # buffer for target v
                    buffer_v_target = []
                    
                    for r in buffer_r[::-1]:
                        v_s_ = r + gamma * v_s_
                        buffer_v_target.append(v_s_)
                        
                    buffer_v_target.reverse()
                    
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                                 self.AC.s: buffer_s,
                                 self.AC.a_his: buffer_a,
                                 self.AC.v_target: buffer_v_target,
                                 }
                    
                    # update global network
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    
                    # get global parameters to local ActorCritic
                    self.AC.pull_global()
                    
                s = s_
                total_step += 1
                if done:
                    if len(global_rewards) < 5:
                        global_rewards.append(ep_r)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] =(np.mean(global_rewards[-5:]))
                    
                    global_episodes += 1
                    break