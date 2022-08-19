import pandas as pd
import os
from collections import deque
from collections.abc import Iterable

import pandas as pd
import wandb

acc_threshold = {
    "google_speech_resnet34": 10,
    "openimage_shufflenet": 10,
    "reddit_albert-base-v2": 10,
    "stackoverflow_albert-base-v2": 10,
    "cifar10_resnet18": 10
}
#
default_rounds = {
    "google_speech_resnet34": 1000,
    "openimage_shufflenet": 1000,
    "reddit_albert-base-v2": 1000,
    "stackoverflow_albert-base-v2": 1000,
    "cifar10_resnet18": 1000
}

def print_full(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def load(project, overwrite=False):
    if not overwrite and os.path.isfile(f'{project}.xlsx'):
        df = pd.read_excel(f'{project}.xlsx', index_col=0)
    else:
        df = pd.DataFrame()
    return df

def load_and_fetch(project, overwrite=False):
    if not overwrite and os.path.isfile(f'{project}.xlsx'):
        df = pd.read_excel(f'{project}.xlsx', index_col=0)
    else:
        df = pd.DataFrame()

    df, parsed_runs = fetch_new_runs(project, df)
    if parsed_runs > 0:
        print(f"updating {project}.xlsx")
        df.to_excel(f'{project}.xlsx')
    else:
        print(f"{project} runs up-to-date")
    return df



def fetch_new_runs(project, df=pd.DataFrame()):
    '''
    return a new df that includes all finished runs, will skip runs already in df and append new runs to df if provided
    '''
    names = set(df.index)
    dfs = [df]
    api = wandb.Api()
    runs = api.runs(f"refl/{project}") #, {"state": "finished"})
    def good_run(run):
        if run.name not in names and len(run.tags)==0:
            if run.state=="finished" or run.state=="crashed":
                return True
            if run.state=='running':
                return False
            df = run.history(samples=2147483647).tail(1)
            if '_step' not in df.columns:
                print(f'{run.url} can be tagged as invalid')
                run.tags.append('no_step_in_col')
                run.update()
                return False
            logged_rounds = df['_step'].values[-1]
            almost = (logged_rounds >= 0.9 * default_rounds[project])
            if almost:
                print(f'including {run.id} with state {run.state} completed {logged_rounds} rounds..')
            else:
                print(f'{run.url} completed too few rounds')
                run.tags.append('too_few_rounds')
                run.update()
            return almost
        return False
   
        
    parsed_runs = [parse_run(run, project) for run in runs if good_run(run)]
    print(f"parsed {len(parsed_runs)} runs for {project}")
    if len(parsed_runs) > 0:
        dfs.append(pd.concat(parsed_runs))
    return pd.concat(dfs), len(parsed_runs)


def parse_run(run, project):
    print(f"parsing {run.name}")
    metrics = ['Test/acc_top_5', 'Test/loss',
               'Round/clock', 'Round/epoch', 'Round/total_updates', 'Round/stale_updates', 'Round/compute', 'Round/communicate']
    #Ahmed - skip if the metrics are not in summary
    for metric in metrics:
        if metric not in run.summary:
            print(f"Wrong Run: {run.url} does not have metric {metric}? skipping")
            return pd.DataFrame()
    #Ahmed - skip if the run has no tags
    if not len(run.tags):
        print(f"{run.url} does not have any tags? skipping")
        return pd.DataFrame()

    df = run.history(samples=2147483647)
    df_last_row = df.dropna(subset=metrics).tail(1)
    if 'minsel' not in run.name and df_last_row['_step'].values[0] < (0.9 * default_rounds[project]):
        print(f"{run.url} finished but has too few rounds? skipping")
        return pd.DataFrame()
    df_last_row[metrics] = df_last_row[metrics].apply(pd.to_numeric, errors='coerce')

    configs = {f'configs/{key}': str(value) if isinstance(value, Iterable) else value for key, value in run.config.items()}
    configs["configs/seed"] = run.name.split('_')[-1]
    configs["configs/config_name"] = '_'.join(run.name.split('_')[:-1])

    configs['tags'] = ','.join(run.tags)
    configs['id'] = run.id
    configs['url'] = run.url
    configs['name'] = run.name
    # configs['heter'] = run.name.split('_')[1] if 'default' in run.name else run.name.split('_')[0]
    # configs['dev_frac'] = f"{configs['configs/low_end_dev_frac']},{configs['configs/moderate_dev_frac']},{configs['configs/high_end_dev_frac']}"
    configs['converged'] = df_last_row[['Test/acc_top_5']].values[0][0] >= acc_threshold[project]
    configs_df = pd.DataFrame(configs, index=df_last_row.index)
    final = pd.concat([configs_df, df_last_row], axis=1, sort=False)
    final.index = [run.name]
    return final


def median_on_seed(df):
    '''
    return median and std based on seeds, with "seed_count" column attached
    '''
    columns = df.columns.tolist()
    group = df.groupby(['configs/config_name', 'configs/partitioning'])
    count = group.size()
    count.name = 'seed_count'
    median = group.median()
    std = group.std()
    median['converged'] = group.agg({'converged': 'any'})
    median['seeds'] = group.agg(seeds=('configs/sample_seed', set))
    median['seed_count'] = count
    std['seed_count'] = count
    return median, std

def test():
    print(1)

def assign_markers(keys, assigned=None):
    markers = deque(["o", "^", "s", "P", "X", "p", "*", "h", "D", "d", "1", "<", ">", "v"])
    if assigned is None:
        assigned = {}
    for key in assigned.values():
        markers.remove(key)
    for key in keys:
        if key in assigned:
            continue
        else:
            assigned[key] = markers.popleft()
    return assigned


def assign_colors(keys, assigned=None):
    colors = deque(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'blue', 'green', 'purple', 'yellow'])
    if assigned is None:
        assigned = {}
    for key in assigned.values():
        colors.remove(key)
    for key in keys:
        if key in assigned:
            continue
        else:
            assigned[key] = colors.popleft()
    return assigned
