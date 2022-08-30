import yaml
import os
import logging
import sys


file = None
yaml_filepath = "experiment_01_run_01.yaml"
save_files = {}

def load_cfg(yaml_filepath):
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.load(stream,Loader=yaml.FullLoader)
    results_dir = cfg['results_facts']['folder']['root_path']
    experiment_number = cfg['experiment_facts']['experiment_number']
    run_number = cfg['experiment_facts']['run_number']
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg,results_dir,experiment_number,run_number)
    return cfg


def make_paths_absolute(dir_, cfg,results_dir,experiment_number,run_number,append=True):
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if append:
                sys.path.append(os.path.dirname(cfg[key]))
            if not os.path.isfile(cfg[key]) or os.path.isdir(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if key.endswith("_dir"):
            save_path = f"{results_dir}/experiment_0{experiment_number}_0{run_number}/{cfg[key]}"
            save_path = os.path.join(dir_,save_path)
            save_path = os.path.abspath(save_path)
            save_files[key] = save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key],results_dir,experiment_number,run_number)
    return cfg


def main():
    path = f"/mnt/d/Playground/Research/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/FashionNeRF/src/experiments/{yaml_filepath}"
    file = load_cfg(path)
    return file
    
file = main()