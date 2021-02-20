import os, sys
import numpy as np
# from ValueNet import ValueModel
# from PolicyNet import PolicyModel
import xml.etree.ElementTree as ET
# import torch
import math as mh


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants
sumoBinary = "/usr/bin/sumo-gui"


if __name__ == "__main__":
    
    config_list = ["config/pingshan.sumocfg"]*int(1e1)
    Epoch=0
    for conf in config_list:
        sumoCmd = [sumoBinary, "-c", conf, "--start"]
        traci.start(sumoCmd)
        traci.gui.setSchema("View #0", "real world")
        
        Epoch += 1
        print("Epoch:", Epoch, conf)
        
        # reset
        kStopEpisode = False
        
        step = 0
        lightlist = traci.trafficlight.getIDList()
        lanelist = traci.lane.getIDList()
        edgelist = traci.edge.getIDList()
        
        # all lanes directly connecting to the lights
        lightentrylanelist = []
        lightexitlanelist = []
        for x in lightlist:
            links = traci.trafficlight.getControlledLinks(x)
            # print(len(links))
            for y in links:
                # if not len(y):
                #     print('ERROR',x, links)
                    
                # if len(y):
                # print(x,y)
                if y[0][0] not in lightentrylanelist:
                    lightentrylanelist.append(y[0][0])
                if y[0][1] not in lightexitlanelist:
                    lightexitlanelist.append(y[0][1])
        
            
        while step < 2e4:
            traci.simulationStep()
            allVList = traci.vehicle.getIDList()
                      
            step += 1
        
        traci.close() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    