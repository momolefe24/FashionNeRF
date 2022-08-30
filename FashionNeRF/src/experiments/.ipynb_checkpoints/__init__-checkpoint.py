import yaml
import os
import pprint
import logging
import sys


file = None
def load_cfg(yaml_filepath):
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.load(stream,Loader=yaml.FullLoader)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg,append=True):
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if append:
                sys.path.append(os.path.dirname(cfg[key]))
            if not os.path.isfile(cfg[key]) or os.path.isdir(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg

def main():
    yaml_filepath = "experiment_01_run01.yaml"
    path = f"/mnt/d/Playground/Research/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/FashionNeRF/src/experiments/{yaml_filepath}"
    file = load_cfg(path)
    return file
    
file = main()