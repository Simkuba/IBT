'''

This file contains functions used for dataset analysis

Author: Jakub Čoček (xcocek00)

'''

import csv
import os
from datetime import timedelta

path_to_module = os.path.abspath(os.path.join('..', 'flowmind'))
sys.path.append(path_to_module)

from flowmind.processing.dataloaders.common import (
    filter_min_flow_length,
    parse_timestamps,
    PPI_PKT_TIMES_KEY,
    PACKETS_KEY,
    PACKETS_REV_KEY,
)

def count_flows(filename: str) -> int:
    '''
    Counts number of flow (number of rows in csv)

    Args:
        filename: path to csv (dataset)

    Returns:
        number of flows

    '''
    flow_count = 0
    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            flow_count += 1
    return flow_count

def count_flows_filtered(filename: str, threshold: int) -> int:
    '''
    Counts number of flow (number of rows in csv) with applied filter

    Args:
        filename: path to csv (dataset)
        threshold: threshold value

    Returns:
        number of flows

    '''
    count = 0
    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if filter_min_flow_length(row, threshold):
                count += 1
    return count 

def calculate_length(x: dict) -> float:
    '''
    Calculates the length of flow

    Args:
        x: one flow

    Returns:
        length of the flow

    '''
    
    times_str = x.get(PPI_PKT_TIMES_KEY, "")
    timestamps = parse_timestamps(times_str)
    if not timestamps:
        return 0
    duration = timestamps[-1] - timestamps[0]
    length = float(duration.total_seconds())
    return length

def calculate_avrg_length(file_path: str) -> float:
    '''
    Calculates the average length of flows in dataset

    Args:
        file_path: path to csv (dataset)

    Returns:
        average length of flows

    '''
    
    lengths = []
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            length = calculate_length(row)
            if length > 0:
                lengths.append(length)
    
    if not lengths:
        return 0.0
    
    return sum(lengths) / len(lengths)

def min_pkts(x: dict, threshold: int) -> bool:
    '''
    Checks for number of packets in flow.

    Args:
        x: one flow
        threshold = threshold value

    Returns:
        True if total packets is more than threshold.

    '''
   
    total_packets = int(x.get(PACKETS_KEY, 0)) + int(x.get(PACKETS_REV_KEY, 0))
    return total_packets > threshold

def calculate_avrg_length_filtered(file_path: str, threshold: int) -> float:
    '''
    Calculates the average length of flows in dataset

    Args:
        file_path: path to csv (dataset)
        threshold: threshold value

    Returns:
        The average length of flows.

    '''
    
    lengths = []
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if min_pkts(row, threshold):
                flow_length = calculate_length(row)
                if flow_length > 0:
                    lengths.append(flow_length)
    
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)

def calculate_avrg_num_pkts(file_path: str, threshold: int) -> float:
    '''
    Calculates the average number of packets in flows in dataset

    Args:
        file_path: path to csv (dataset)
        threshold: threshold value

    Returns:
        The average number of packets.

    '''
    total_flows = 0
    total_pkts  = 0

    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pkts = int(row.get(PACKETS_KEY, 0)) + int(row.get(PACKETS_REV_KEY, 0))
            if pkts < threshold:
                continue

            total_flows += 1
            total_pkts  += pkts

    if total_flows == 0:
        return 0.0
    return total_pkts / total_flows
