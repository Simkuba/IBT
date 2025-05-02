'''

This file contains custom version of FlowMind dataloader and support functions.

'''

# -- IMPORTS --

import os
os.chdir('/workplace/flowmind/')

# torch imports
import torch
from torchdata.datapipes.iter import FileLister

# flowmind imports
from flowmind.processing.dataloaders.common import FlowData
from flowmind.processing.dataloaders.flowpic import Flowpic
from flowmind.processing.dataloaders.common import (
    DataPipeTransform,
    FlowDataTransform,
    datapipe_identity,
    filter_csv_filename,
    filter_min_flow_length,
    flowdata_identity,
)
from flowmind.processing.scaler import Scaler

# others
from functools import partial
from pathlib import Path
from datetime import timedelta

def calculate_length(flow: FlowData, threshold: int) -> bool:
    '''
    Calculates length of the flow and return True or False if the flow 
    is above threshold or not.

    Args:
        flow: The flow in FlowData format.
        threshold: Threshold.

    Returns:
        True if the length is at least the threshold value. False otherwise.
    '''
    return (flow.times[-1] - flow.times[0]) >= timedelta(threshold)

# -- custom dataloader creating function --
def create_flowpic_dataloader(
    dir_path: Path | str,
    batch_size: int,
    scaler: Scaler | None = None,
    min_packets: int = 0,
    time_bins: list = [0, 37.5, 75.0, 112.5, 150.0, 187.5, 225.0, 262.5, 300.0],
    length_bins: list = [0, 187.5, 375.0, 562.5, 750.0, 937.5, 1125.0, 1312.5, 1500.0],
    ppi_bins: list | None = None,
    normalize: bool = False,
    bidirectional: bool = True,
    meta_key: str | list[str] | None = None,
    dp_transform: DataPipeTransform = datapipe_identity,
    flow_transform_1: FlowDataTransform = flowdata_identity, 
    flow_transform_2: FlowDataTransform = flowdata_identity, 
    num_workers: int = 0,
    min_length: int = 0,
) -> torch.utils.data.DataLoader:
    """Create torch DataLoader.
       The dataloader creates 2d flowpic from ppi and transform them into tensors.

    Args:
        dir_path (Path | str): Directory with csvs.
        batch_size (int): Batch size.
        scaler (Mapping[str, Scaler] | None, optional): Data scaler to apply. Defaults to None.
        min_packets (int, optional): Minimal flow lengths to include. Defaults to 0.
        time_bins (list, optional): Time bins. Defaults to [0, 37.5, 75.0, 112.5, 150.0, 187.5, 225.0, 262.5, 300.0].
        length_bins (list, optional): Packet size bins. Defaults to [0, 187.5, 375.0, 562.5, 750.0, 937.5, 1125.0, 1312.5, 1500.0].
        ppi_bins (list | None, optional): PPI bins. Defaults to None.
        normalize (bool, optional): Whether to normalize packets in bins instead of absolute counts. Defaults to False.
        bidirectional (bool, optional): Whether to use bidirectional flowpic. Defaults to False.
        meta_key (str | list[str] | None, optional): Target column name. Defaults to None.
        dp_transform(DataPipeTransform, optional): Datapipe transform function. Defaults to identity.
        flow_transform1(FlowDataTransform, optional): Flow transform function. Defaults to identity.
        flow_transform2(FlowDataTransform, optional): Flow transform function. Defaults to identity.

    Returns:
        torch.utils.data.DataLoader: Torch DataLoader.
    """

    # pipeline
    dp = (
        FileLister(str(dir_path))
        .filter(filter_csv_filename)
        .sharding_filter() 
        .open_files() 
        .parse_csv_as_dict()
        .filter(partial(filter_min_flow_length, threshold=min_packets)) 
        .filter(partial(calculate_length, threshold=min_length))
        .map(lambda row: (row, row)) # each item in dp is (row, row)
        .map(
            partial(
                build_two_augmented_flows,
                flow_transform_1=flow_transform_1,
                flow_transform_2=flow_transform_2,
                time_bins=time_bins,
                length_bins=length_bins,
                ppi_bins=ppi_bins,
                normalize=normalize,
                bidirectional=bidirectional,
                meta_key=meta_key,
                scaler=scaler,
            )
        )
    )

    # for label transform (in this work)
    dp = dp_transform(dp)

    return torch.utils.data.DataLoader(dp.batch(batch_size).collate(), batch_size=None, num_workers=num_workers)

def build_two_augmented_flows(
    pair_of_rows: tuple[dict, dict],
    flow_transform_1,
    flow_transform_2,
    time_bins,
    length_bins,
    ppi_bins,
    normalize,
    bidirectional,
    meta_key,
    scaler=None,
):
    """
    Creates (flowpic1, flowpic2, label) from (row1, row2).

    Args:
        pair_of_rows: tuple of duplicated rows (views)
        scaler (Mapping[str, Scaler] | None, optional): Data scaler to apply. Defaults to None.
        time_bins (list, optional): Time bins. Defaults to [0, 37.5, 75.0, 112.5, 150.0, 187.5, 225.0, 262.5, 300.0].
        length_bins (list, optional): Packet size bins. Defaults to [0, 187.5, 375.0, 562.5, 750.0, 937.5, 1125.0, 1312.5, 1500.0].
        ppi_bins (list | None, optional): PPI bins. Defaults to None.
        normalize (bool, optional): Whether to normalize packets in bins instead of absolute counts. Defaults to False.
        bidirectional (bool, optional): Whether to use bidirectional flowpic. Defaults to False.
        meta_key (str | list[str] | None, optional): Target column name. Defaults to None.
        flow_transform1(FlowDataTransform, optional): Flow transform function. Defaults to identity.
        flow_transform2(FlowDataTransform, optional): Flow transform function. Defaults to identity.

    Returns:
        FlowPic1, FlowPic2 and corresponding label.
    """
    row1, row2 = pair_of_rows

    # first flow
    flow1 = Flowpic(
        x=row1,
        flow_transform=flow_transform_1,
        time_bins=time_bins,
        length_bins=length_bins,
        ppi_bins=ppi_bins,
        normalize=normalize,
        bidirectional=bidirectional,
        meta_key=meta_key,
    )
    out1 = flow1.export(scaler=scaler)  # => (tensor, label)

    # second flow
    flow2 = Flowpic(
        x=row2,
        flow_transform=flow_transform_2,
        time_bins=time_bins,
        length_bins=length_bins,
        ppi_bins=ppi_bins,
        normalize=normalize,
        bidirectional=bidirectional,
        meta_key=meta_key,
    )
    out2 = flow2.export(scaler=scaler)

    # remove duplicate label
    fp1, label = out1    
    fp2, _ = out2

    return fp1, fp2, label