#!/usr/bin/env python
# coding: utf-8


import math
import numpy as np


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []  
        self.debug = debug
        self.time_span = time_span  
        self.time_interval = time_interval 
        
    def append(self,obj):  
        self.objects.append(obj)
    
    def dist(self): 
        elems = {}
        for i in range(int(self.time_span/self.time_interval)+1):
            self.one_step(i, elems)
        self.elems = elems
        
    def one_step(self, i, elems):
        time = self.time_interval*i
        for obj in self.objects:
            if hasattr(obj, 'dist'):obj.dist(elems, time)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)    


class IdealRobot:   
    def __init__(self, pose, agent=None, sensor=None): 
        self.pose = pose 
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor 
         
    @classmethod           
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0), 
                                    nu*math.sin(t0), 
                                    omega])*time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )

    def one_step(self, time_interval):
        if not self.agent: return        
        obs =self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        if self.sensor: self.sensor.data(self.pose)   

    def dist(self, elems, time):
        if self.agent and hasattr(self.agent, "dist"):    
            self.agent.dist(elems, time)
        return
    
class Agent: 
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
        
    def decision(self, observation=None):
        return self.nu, self.omega


class Map:
    def __init__(self):       
        self.landmarks = []