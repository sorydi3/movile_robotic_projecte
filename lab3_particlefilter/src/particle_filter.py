#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from probabilistic_lib.functions import angle_wrap, get_polar_line
import math
import random


#===============================================================================
class ParticleFilter(object):
    '''
    Class to hold the whole particle filter.
    
    p_wei: weights of particles in array of shape (N,)
    p_ang: angle in radians of each particle with respect of world axis, shape (N,)
    p_xy : position in the world frame of the particles, shape (2,N)
    '''
    
    #===========================================================================
    def __init__(self, room_map, num, odom_lin_sigma, odom_ang_sigma, 
                 meas_rng_noise, meas_ang_noise,x_init,y_init,theta_init):
        '''
        Initializes the particle filter
        room_map : an array of lines in the form [x1 y1 x2 y2]
        num      : number of particles to use
        odom_lin_sigma: odometry linear noise
        odom_ang_sigma: odometry angular noise
        meas_rng_noise: measurement linear noise
        meas_ang_noise: measurement angular noise
        '''
        
        # Copy parameters
        self.map = room_map
        self.num = num
        self.odom_lin_sigma = odom_lin_sigma
        self.odom_ang_sigma = odom_ang_sigma
        self.meas_rng_noise = meas_rng_noise
        self.meas_ang_noise = meas_ang_noise
        self.vehicle_frame=False
        # Map
        map_xmin = np.min(self.map[:, 0])
        map_xmax = np.max(self.map[:, 0])
        map_ymin = np.min(self.map[:, 1])
        map_ymax = np.max(self.map[:, 1])
        
        # Particle initialization arround starting point
        self.p_wei = 1.0 / num * np.ones(num)
        self.p_ang =2 * np.pi * np.random.rand(num)
        self.p_xy  = np.vstack(( x_init+ 1*np.random.rand(num)-0.5,
                                 y_init+ 1*np.random.rand(num)-0.5 ))
        #Flags for resampling                         
        self.moving=False
        self.n_eff=0 #Initialize Efficent number as 0

        
    
    #===========================================================================
    def predict(self, odom):
        '''
        Moves particles with the given odometry.
        odom: incremental odometry [delta_x delta_y delta_yaw] in the vehicle frame
        '''
        #Check if we have moved from previous reading.
        if odom[0]==0 and odom[1]==0 and odom[2]==0:
            self.moving=False
        else:
            
            # TODO: code here!!
            # Add Gaussian noise to odometry measures
            # .... np.random.randn(...)
            #
            x = odom[0]
            y=odom[0]
            h=odom[2]

            #transform particles from world frame to vehicle frame.

            for i in range(len(self.p_xy[0,:])):
                soroll_x=np.random.normal()*self.odom_lin_sigma
                soroll_y=np.random.normal()*self.odom_lin_sigma
                soroll_h=np.random.normal()*self.odom_ang_sigma
                self.p_xy[0,i]+=x*np.cos(self.p_ang[i])+soroll_x
                self.p_xy[1,i]+=y*np.sin(self.p_ang[i])+soroll_y
                self.p_ang[i]+= angle_wrap(h+soroll_h)
            # Increment particle positions in correct frame
            #self.p_xy +=
            #
            #
            #

            # Increment angle
            #self.p_ang += ...
            #self.p_ang = angle_wrap(self.p_ang)
            
            #Update flag for resampling
            self.moving=True     
    
    #===========================================================================
    def weight(self, lines):
        '''
        Look for the lines seen from the robot and compare them to the given map.
        Lines expressed as [x1 y1 x2 y2].
        '''
        # TODO: code here!!
        # Constant values for all weightings
        val_rng = 1.0 / (self.meas_rng_noise * np.sqrt(2 * np.pi))
        val_ang = 1.0 / (self.meas_ang_noise * np.sqrt(2 * np.pi))
        
        # Loop over particles
        for i in range(self.num):
            #
            # Transform map lines to local frame and to [range theta]
            #
            map_to_polar=np.zeros((self.map.shape[0],2))
            for j in range(self.map.shape[0]):
                map_to_polar[j,:]=get_polar_line(self.map[j],[self.p_xy[0,i],self.p_xy[1,i],self.p_ang[i]])
                
            # Transform measured lines to [range theta] and weight them
            #
            lines_to_polar=np.zeros((lines.shape[0],2))
            weight_lines = np.zeros((self.map.shape[0],lines.shape[0]))
            for j in range(lines.shape[0]):
                #
                lines_to_polar[j,:]=  get_polar_line(lines[j],[0,0,0])
                #
                # Weight them
                weight_lines_range = val_rng * np.exp(-np.power(lines_to_polar[j,0] - map_to_polar[:,0],2) / (2*np.power(self.meas_rng_noise,2))) 
                weight_lines_angle = val_ang * np.exp(-np.power(lines_to_polar[j,1] - map_to_polar[:,1],2) / (2*np.power(self.meas_ang_noise,2))) 
                weight_lines[:,j] = weight_lines_angle * weight_lines_range
                #
                #
                #
                #
                
            # OPTIONAL question
            # make sure segments correspond, if not put weight to zero
            #
            #
            
            # Take best weighting (best associated lines)
            #
            max_weight = np.amax(weight_lines,axis = 0)
            for j in range(max_weight.shape[0]):
                self.p_wei[i] = self.p_wei[i] * max_weight[j]
            
        # Normalize weights
        self.p_wei /= np.sum(self.p_wei)
        # TODO: Compute efficient number
        self.n_eff= 1/(np.sum(np.power(self.p_wei,2)))
        
    #===========================================================================
    def resample(self):
        '''
        Systematic resampling of the particles.
        '''
        # TODO: code here!!
        # Look for particles to replicate
        r = random.uniform(0, 1./self.num)
        c = self.p_wei[0]
        i = 0
        for m in range(self.num):
            u = r + m * 1./self.num
            while u > c :
                i = i + 1
                c += self.p_wei[i]

            self.p_xy[:,m] = self.p_xy[:,i]
            self.p_ang[m] = self.p_ang[i]

        self.p_wei = 1.0 / self.num * np.ones(self.num)
    
    #===========================================================================
    def get_mean_particle(self):
        '''
        Gets mean particle.
        '''
        # Weighted mean
        weig = np.vstack((self.p_wei, self.p_wei))
        mean = np.sum(self.p_xy * weig, axis=1) / np.sum(self.p_wei)
        
        ang = np.arctan2( np.sum(self.p_wei * np.sin(self.p_ang)) / np.sum(self.p_wei),
                          np.sum(self.p_wei * np.cos(self.p_ang)) / np.sum(self.p_wei) )
                          
        return np.array([mean[0], mean[1], ang])
