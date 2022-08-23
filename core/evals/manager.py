# Submit job to the remote cluster
import datetime
import os
import pickle
import random
import subprocess
import sys
import time
from os import walk
from pathlib import Path

from envyaml import EnvYAML


def load_yaml_conf(yaml_file):
    data = EnvYAML(yaml_file)
    return data

def process_cmd(yaml_file, node_ip, num_gpus):

    wandb_key = os.environ['WANDB_API_KEY']
    yaml_conf = load_yaml_conf(yaml_file)
    #print(yaml_conf)

    gpu_ids = []
    for i in range(int(num_gpus)):
        gpu_ids.append(i)

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    job_name = 'kuiper_job'
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp':time_stamp,
                'ps_ip':node_ip,
                'ps_port':random.randint(1000, 60000),
                'manager_port':random.randint(1000, 60000)
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)
    print(job_conf)

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "
    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = job_conf[conf_name]
    log_path = os.path.join(log_path, 'logs', job_name, time_stamp)
    print(f'job: {job_name} log: {log_path}')
    Path(log_path).mkdir(parents=True, exist_ok=True)

    learner_conf = '-'.join([str(_) for _ in list(range(1, int(num_gpus)+1))])
    # =========== Submit job to parameter server ============
    running_vms.add(node_ip)
    #Ahmed - Note we are not using a GPU for the aggreagtor by setting use_cuda=0
    ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --learner={learner_conf} --use_cuda=0"

    with open(f"{log_path}/all_logs", 'wb') as fout:
        pass

    print(f"Starting aggregator on {node_ip}...")
    with open(f"{log_path}/all_logs", 'a') as fout:
        subprocess.Popen(f'ssh {submit_user}{node_ip} "{setup_cmd} {ps_cmd}"', shell=True, stdout=fout, stderr=fout)
    print(f'ssh {submit_user}{node_ip} "{setup_cmd} {ps_cmd}"')

    time.sleep(20)
    # =========== Submit job to each worker ============
    worker_ip = node_ip
    running_vms.add(worker_ip)
    for gpu in gpu_ids:
        print(f"Starting workers on {worker_ip}:{gpu} ...")
        worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={gpu+1} --learner={learner_conf} --cuda_device=cuda:{gpu} "
        with open(f"{log_path}/all_logs", 'a') as fout:
                subprocess.Popen(f'ssh {submit_user}{worker_ip} "{setup_cmd} {worker_cmd}"', shell=True, stdout=fout, stderr=fout)
        print(f'ssh {submit_user}{node_ip} "{setup_cmd} {worker_cmd}"')

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    Path(os.path.join(current_path, 'job_locations', job_name)).mkdir(parents=True, exist_ok=True)
    job_meta_path = os.path.join(current_path, 'job_locations', job_name, time_stamp)
    with open(job_meta_path, 'wb') as fout:
        job_meta = {'user':submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs ({log_path}) for status")

def terminate(job_name, time_stamp = ''):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, 'job_locations', job_name, time_stamp)
    job_files = []
    time_stamps = []
    job_names = []
    if time_stamp != '':
        job_files.append(os.path.join(current_path, 'job_locations', job_name, time_stamp))
        job_names.append(job_name)
        time_stamps.append(time_stamp)
    else:
        if job_name != 'all':
            job_folder = os.path.join(current_path, 'job_locations', job_name)
            for (dirpath, dirnames, filenames) in walk(job_folder):
                for file in filenames:
                    job_files.append(os.path.join(job_folder, file))
                    time_stamps.append(file)
                    job_names.append(job_name)
                break
        else:
            main_folder = os.path.join(current_path, 'job_locations')
            for job_name in os.listdir(main_folder):
                job_folder = os.path.join(main_folder, job_name)
                for (dirpath, dirnames, filenames) in walk(job_folder):
                    for file in filenames:
                        job_files.append(os.path.join(job_folder, file))
                        time_stamps.append(file)
                        job_names.append(job_name)
                    break

    with open(f"shutdown_logging", 'w') as fout:
        pass

    if len(job_files):
        for i, job_meta_path in enumerate(job_files):
            with open(job_meta_path, 'rb') as fin:
                job_meta = pickle.load(fin)
            for vm_ip in job_meta['vms']:
                print(f"Shutting down job on {vm_ip} in {job_meta_path}")
                with open(f"shutdown_logging", 'a') as fout:
                    subprocess.Popen(f'ssh {job_meta["user"]}{vm_ip} "python {current_path}/shutdown.py {job_names[i]} {time_stamps[i]}"', shell=True, stdout=fout, stderr=fout)
                time.sleep(2)
            os.system("rm {}".format(job_meta_path))

if sys.argv[1] == 'submit':
    process_cmd(sys.argv[2], sys.argv[3], sys.argv[4])
elif sys.argv[1] == 'stop':
    if len(sys.argv) > 3:
        terminate(sys.argv[2], sys.argv[3])
    else:
        terminate(sys.argv[2], '')
else:
    print("Unknown cmds ...")

