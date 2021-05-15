import subprocess
# import json
import argparse
import re
import importlib
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='train config file path')
args = parser.parse_args()
config_file=args.config

with open(config_file) as f:
    orig_lines = f.readlines()
# /opt/ml/code/mmdetection_trash/configs/_custom_config/swin.py -> /opt/ml/code/mmdetection_trash/configs/_custom_config/siwn0.py
for idx in range(5):
    new_config=config_file[:-3]+str(idx)+config_file[-3:]
    # print(new_config)
    lines=orig_lines[:]
    with open(new_config, "w") as f:
        for i,line in enumerate(lines):
            if 'ann_file' in line and 'test' not in line:
                # /opt/ml/input/data/train_data0.json
                lines[i]=line.replace('.json', f'_data{idx}.json')
            elif 'name' in line:
                # line=line+'_'+str(idx)
                coms = line.split("\'")
                coms[1] = f"{coms[1]}_{idx}"
                lines[i]="\'".join(coms)
        f.write("".join(lines))
        # break
    a=subprocess.Popen(["python", "tools/train.py",new_config,'--work-dir','./work_dirs/'+f"{config_file.split('/')[-1][:-3]}_kold/{new_config.split('/')[-1][:-3]}"], stdout=subprocess.PIPE)
    a.communicate() 