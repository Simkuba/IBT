'''

This file contains functions for data (flow) augmentations.

Author: Jakub Čoček (xcocek00)

'''

# -- IMPORTS --

import os
os.chdir('/workplace/flowmind/')

# torch imports
import torchvision.transforms as T

# flowmind imports
from flowmind.processing.dataloaders.common import FlowData

# others
from datetime import timedelta
import random

# -- RTT augmentation --
def augment_rtt(flow: FlowData, alpha_min: float = 0.5, alpha_max: float = 1.5) -> FlowData:
    """
    Multiply arrival time of each packet by a factor alpha, where
    alpha is chosen uniformly in [alpha_min, alpha_max]

    Args
        flow: original flow
        alpha_min: min factor set to 0.5
        alpha_max: max factor set to 1.5

    Returns: 
        modified FlowData
    """
    
    if not flow.times:
        return flow

    # select random alpha 
    alpha = random.uniform(alpha_min, alpha_max)
    init_time = flow.init_time

    # convert time to float offset
    offsets = [(t - init_time).total_seconds() for t in flow.times]
    
    # RTT augmentation
    offsets = [offset * alpha for offset in offsets]

    # convert back
    flow.times = [init_time + timedelta(seconds=offset) for offset in offsets]

    return flow

# -- IAT augmentation --
def augment_iat(flow: FlowData, b_min: float = -1.0, b_max: float = 1.0) -> FlowData:
    '''
    Add factor b to the arrival time of each packet, where
    b is chosen uniformly in [b_min, b_max]

    Args:
        flow: original flow
        b_min: min factor set to -1.0 
        b_max: max factor set to 1.0

    Returns: 
        modified FlowData
    '''
    
    if not flow.times:
        return flow
    
    # select random b 
    b = random.uniform(b_min, b_max)
    init_time = flow.init_time

    # convert time to float offset
    offsets = [(t - init_time).total_seconds() for t in flow.times]
    
    # IAT augmentation
    offsets = [offset + b for offset in offsets]

    # convert back
    flow.times = [init_time + timedelta(seconds=offset) for offset in offsets]

    return flow

# -- packet loss augmentation --
def packet_loss(flow: FlowData, dt: float = 0.1) -> FlowData:
    '''
    Packet loss augmentation - removing all packets within [t - dt, t + dt]
    
    Args:
        flow: original flow
        dt: delta t (interval length) set to 0.1
    
    Returns: 
        modified Flow with removed packets in that interval
    ''' 

    if not flow.times:
        return flow
    
    init_time = flow.init_time 
    
    # conver to offset
    offsets = [(t - init_time).total_seconds() for t in flow.times]
    flow_duration = offsets[-1] if offsets else 0.0

    # zero or near-zero duration -> skip (would remove the whole flow)
    limit = 2*dt
    if flow_duration <= limit:
        return flow

 
    t = random.uniform(0, flow_duration)

    # interval [t - dt, t + dt]
    lower = t - dt
    upper = t + dt

    # lists for filtered packets
    new_directions = []
    new_lengths    = []
    new_times      = []
    new_push_flags = []

    for offset, direction, length, time_val, push_flag in zip(offsets, flow.directions, flow.lengths, flow.times, flow.push_flags):
        if not (lower <= offset <= upper):
            new_directions.append(direction)
            new_lengths.append(length)
            new_times.append(time_val)
            new_push_flags.append(push_flag)

    # replace old lists with the fitered lists
    flow.directions = new_directions
    flow.lengths    = new_lengths
    flow.times      = new_times
    flow.push_flags = new_push_flags
    flow.init_time  = flow.times[0]

    return flow

# -- FlowPic rotation --
def flowpic_rotation(flowpic):
    '''
    FlowPic rotation augmentation - rotating flowpic within -10° and 10°

    Args:
        flowpic: original flowpic

    Returns:
        rotated flowpic
    '''
    rotate_transform = T.RandomRotation(degrees=(-10,10))
    return rotate_transform(flowpic)