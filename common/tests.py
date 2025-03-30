# -- IMPORTS --

import os
os.chdir('/workplace/flowmind/')

# torch imports
import torch

# others
import sys
import csv

# sets csv limit
csv.field_size_limit(sys.maxsize)

# dimensions validation
def dim_val(dl_train: torch.utils.data.DataLoader) -> None:
    print("dl train: ", type(dl_train), file=sys.stderr)
    for (flowpic1, flowpic2, labels) in dl_train:
        print("flowpic1", type(flowpic1), file=sys.stderr)
        print("flowpic2", type(flowpic2), file=sys.stderr)
        print("flowpic1 shape: ", flowpic1.shape, file=sys.stderr)
        print("flowpic2 shape: ", flowpic2.shape, file=sys.stderr)
        break    

# print batch
def print_batch(dl_train: torch.utils.data.DataLoader, type: str) -> None:
    for batch in dl_train:
        torch.set_printoptions(threshold=sys.maxsize)
        print("printing " + type + " batch ...,", file=sys.stderr)
        print(batch, file=sys.stderr)
        break
    
# debug batch    
def debug_batch(dl: torch.utils.data.DataLoader) -> None:
    for (flowpic1, flowpic2, labels) in dl:
        print("flowpic1 type:", type(flowpic1), file=sys.stderr)
        print("flowpic2 type:", type(flowpic2), file=sys.stderr)
        print("label type:", type(labels), file=sys.stderr)
        
        if isinstance(flowpic1, list):
            print(f"flowpic1 has {len(flowpic1)} items.", file=sys.stderr)
            print("Shapes of flowpic1 items:", file=sys.stderr)
            for i, fp in enumerate(flowpic1):
                print(type(fp), file=sys.stderr)
                print(f"  Item {i}: {fp.shape}", file=sys.stderr)
                break

        if isinstance(flowpic2, list):
            print(f"flowpic2 has {len(flowpic2)} items.", file=sys.stderr)
            print("Shapes of flowpic2 items:", file=sys.stderr)
            for i, fp in enumerate(flowpic2):
                print(f"  Item {i}: {fp.shape}", file=sys.stderr)

        print("labels:", labels, file=sys.stderr)
        break  # Just inspect the first batch
    

