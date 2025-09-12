import ray,os,json,random
import networkx as nx
from os.path import join
import numpy as np
import os
import ray
from os.path import join
from collections import Counter
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from typing import cast
from sklearn.model_selection import train_test_split

ray.init(_plasma_directory="/tmp")
def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Config File",
                            default="config.yaml",
                            type=str)
    return arg_parser


@ray.remote
def read_xfg(XFG_root_path,XFG_path):
        return [XFG_path,nx.read_gpickle(join(XFG_root_path,XFG_path)).graph["label"]]

def return_data(datas,index):
    arr=[]
    for  data in datas:
        arr.append(data[index])
    return arr
        
def balance_data(config):
    XFG_root_path=join(os.environ["SLURM_TMPDIR"],"XFG")
    out_root_path=join(config.root_folder_path,config.split_folder_name)
    with open(join(json_root_path,"train.json"), "r") as f:
                testcaseids1=list(json.load(f))
    with open(join(json_root_path,"val.json"), "r") as f:
                testcaseids2=list(json.load(f))
    xfgs=[]
    testcaseids=testcaseids1+testcaseids2
    for j in testcaseids:
            xfgs.append(read_xfg.remote(XFG_root_path,j))
    results = ray.get(xfgs)
    source_code_filtered_neg=[]
    data=[]
    for index,xfg in enumerate(results):
        if xfg[1]==0:
            source_code_filtered_neg.append(xfg)
        else:
            data.append(xfg)
    data.extend(random.sample(source_code_filtered_neg,len(data)))

    
    train_X, val_X = train_test_split(data, test_size=0.1)
    
    X_train=return_data(train_X,0)
    X_val=return_data(val_X,0)
    
    if not os.path.exists(f"{out_root_path}"):
        os.makedirs(f"{out_root_path}")
    with open(f"{out_root_path}/train.json", "w") as f:
        json.dump(X_train, f)
    with open(f"{out_root_path}/val.json", "w") as f:
        json.dump(X_val, f)
    
    
if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config_path=__args.config
    config = cast(DictConfig, OmegaConf.load(config_path))
    balance_data(config)