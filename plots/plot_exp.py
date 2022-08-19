import math
import sys
from pathlib import Path

import matplotlib.colors as matcolors
from matplotlib import pylab as plt

from utils import *

mcolors = matcolors.BASE_COLORS

barwidth = 0.28
barspace = 0.03

# plt.style.use('seaborn-pastel')
SCATTER_SIZE = 50
FONT_SIZE = 30
fig_width_pt = 800  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
fig_width = fig_width_pt * inches_per_pt  # width in inches
golden_mean = (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {'backend': 'pdf',  # 'ps',
          'axes.labelsize': FONT_SIZE,
          'font.size': FONT_SIZE,
          'legend.fontsize': FONT_SIZE,
          'legend.loc': 'upper left',  # 'best', #'upper left',
          'xtick.labelsize': FONT_SIZE,
          'ytick.labelsize': FONT_SIZE,
          # 'text.usetex': True,
          'legend.handlelength': 2,
          'lines.linewidth': 4,
          'figure.figsize': fig_size,
          # 'pdf.fonttype': 42,
          }
plt.rcParams.update(params)

fig_names = {
    'random_fedavg': 'random_fedavg',
    'oort_fedavg': 'oort_fedavg',
    'random_prox': 'random_prox',
    'oort_prox': 'oort_prox',
    'random_yogi': 'random_yogi',
    'oort_yogi': 'oort_yogi',
}

fmts = {
    fig_names['random_fedavg']: '-b',
    fig_names['oort_fedavg']: '-r',

    fig_names['random_yogi']: ':c',
    fig_names['oort_yogi']: ':g',

    fig_names['random_prox']: '--m',
    fig_names['oort_prox']: '--y',
}

linestyles = {
    'fedavg': '-',
    'prox': ':',
    'yogi': '--'
}

colors = {
    'OORT': 'tab:orange',
    'Oort': 'tab:orange',
    'Oort+All': 'tab:orange',
    'Oort+Dyn': 'tab:pink',
    'SAFA': 'tab:purple',
    'SAFA+O': 'tab:purple',
    'RELAY': 'tab:blue',
    'Priority': 'tab:green',
    'RELAY+APT': 'tab:green',
    'FedAvg_10': 'tab:red',
    'FedAvg_100': 'tab:red',
    'Random': 'tab:red',
    'Random+All': 'tab:red',
    'Random+Dyn': 'tab:brown',
    'Equal': 'tab:orange',
    'DynSGD': 'tab:green',
    'AdaSGD': 'tab:red'
}

markers = {
    'stale_update_0': 'o',
    'stale_update_2': 's',
    'stale_update_-1': '*',
    'stale_update_5': '^'
}

main_dir = './plots'

api = wandb.Api()
proj_names = ['google_speech_resnet34', 'cifar10_resnet18']
exp_types = [1]
sample_methods = ['random', 'oort']
grad_policy = ['FedAvg', 'Prox', 'YoGi']
deadline = [-1, -2]
max_epoch = 900
stale_update = [0, 2, -1]
total_workers = [10, 50, 100]
seeds = [0, 1, 2]
percent = {0: 1, 25: 2, 75: 3, 100: 4}
factor = {0: 'NoStale', 1: 'Equal', -2: 'DynSGD', -3: 'AdaSGD', -4: 'RELAY'}

kernel_size = 5


def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)


def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if height > 0.1:
            continue
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                '%0.2f' % height,
                ha='center', va='bottom', fontsize=14, color='red')


def smooth(y, box_pts=1):
    if method.startswith('safa'):
        return y

    cumsum_vec = np.cumsum(np.insert(y, 0, 0))
    y_smooth = (cumsum_vec[box_pts:] - cumsum_vec[:-box_pts]) / float(box_pts)
    # print(len(y), len(y_smooth))
    if len(y) > len(y_smooth):
        avg = np.average(y[len(y_smooth):])
        for i in range(len(y_smooth), len(y)):
            y_smooth = np.insert(y_smooth, i, avg)
      return y_smooth


def get_dfs(project, exp_type, total_worker):
    global method, tags, metric, rand_behv, clients, sampling_method

    print('handling:', project, metric, exp_type, total_worker)

    runs = api.runs('refl/' + project)

    check_tags = []
    if tags != '':
        check_tags.extend(str(tags).split('_'))
    print('inputs: ', project, exp_type, total_worker, check_tags)
    t_runs = {}

    if method.startswith('safa'):
        for run in runs:
            if 'hidden' in run.tags or 'wrong' in run.tags or 'out' in run.tags or tags not in run.tags:
                continue
            if run.state == 'running' or run.state == 'killed':
                continue
            if clients > 0 and run.config['total_clients'] != clients:
                continue
            # Ahmed - exclude infinity for stale updates
            if 'stale_update' in run.config and run.config['stale_update'] == -1:
                continue
            if 'safa' in run.tags or 'r-safa' in run.tags or 'n-safa' in run.tags or 'safa1' in run.tags:
                if run.config['partitioning'] not in t_runs:
                    t_runs[run.config['partitioning']] = []
                t_runs[run.config['partitioning']].append(run)
                print('Add: ', run.tags, run, run.name)
    else:
        for run in runs:

            if '1node' in run.tags or 'adapt' in run.tags:
                continue
            if 'hidden' in run.tags or 'wrong' in run.tags or 'out' in run.tags or run.state == 'running':  # or run.state == 'killed':
                continue
            if len(check_tags) > 0:
                if len(run.tags) <= 0:
                    continue
                if restrict and len(run.tags) != len(check_tags):
                    continue
                else:
                    all_ok = True
                    for tag in check_tags:
                        if tag not in run.tags:
                            all_ok = False
                            break
                    if not all_ok:
                        continue
            if 'selectadapt' not in tags and 'Round/epoch' in run.summary and run.summary['Round/epoch'] < max_epoch:
                print('REDO: ', run.summary['Round/epoch'], run.url)
                continue
            # if method != 'adapt_select' and run.config['adapt_selection'] == 0:
            #     continue
            if tags == 'avail_yogi' and 'converge' in run.tags:
                continue
            if 'converge' in run.tags and (
                    (run.config['sample_mode'] == 'random' and run.config['stale_update'] == 0) or (
                    run.config['stale_update'] == 0 and run.config['avail_priority'] == 1)):
                continue

            if method == 'adapt_select' and (run.config['stale_update'] == 0 or run.config['sample_seed'] == 0):
                print('Skip `: ', run.name)
                continue
            if 'google' in project and 'scale' not in tags and 'yogi' in check_tags and run.config[
                'stale_update'] != 0 and (run.config['adapt_selection'] == 1 or run.config['stale_factor'] != -4):
                print('Skip 2: ', run.name)
                continue
            if 'selectadapt' in 'tags' and run.config['total_worker'] == 100:
                continue
            if ('stale_yogi' == tags or 'stale_selectadapt' in tags) and run.config['sample_mode'] == 'oort' and \
                    run.config['stale_update'] != 0:
                print('Skip 3: ', run.name)
                continue
            # if 'openimg' in project and ('avail_yogi' == tags and run.config['stale_update'] != 0 and run.config['adapt_selection'] == 1):
            #     continue
            # if 'fedavg' in check_tags and run.config['sample_seed'] == 0:
            #    continue
            if 'exp_type' in run.config and run.config['exp_type'] != exp_type:
                print('Skip 4: ', run.name)
                continue
            if rand_behv is not None and 'rand_behv' in run.config and run.config['rand_behv'] != rand_behv:
                print('Skip 5: ', run.name)
                continue
            if train_ratio is not None and 'train_ratio' in run.config and float(run.config['train_ratio']) != float(
                    train_ratio):
                print('Skip 6: ', run.name)
                continue
            if 'stale_factor' in run.config and int(run.config['stale_factor']) not in factor.keys():
                print('Skip 8: ', run.name)
                continue

            if sampling_method is not None:
                if sampling_method == 'relay':
                    if run.config['stale_update'] == 0:
                        continue
                elif ('sample_mode' in run.config and run.config['sample_mode'] != sampling_method) or run.config[
                    'stale_update'] != 0:
                    continue
            if total_worker != 0 and 'total_worker' in run.config and run.config['total_worker'] != total_worker:
                continue
            if tags == 'plot_avail' and run.config['stale_update'] != 0 and run.config['avail_priority'] == 0:
                continue
            if tags == 'plot_scalesyspercent' and run.config['scale_sys_percent'] == 0.5:
                continue
            if run.config['partitioning'] not in t_runs:
                t_runs[run.config['partitioning']] = []
            t_runs[run.config['partitioning']].append(run)
            if 'scale_sys_percent' in run.config:
                print('Add: ', run.config['partitioning'], run.config['scale_sys_percent'],
                      run.config['adapt_selection'], run.config['stale_update'], run.config['stale_factor'], run.tags,
                      run, run.name)
            else:
                print('Add: ', run.config['partitioning'], run.config['adapt_selection'], run.config['stale_update'],
                      run.config['stale_factor'], run.tags, run, run.name)

    fin_epoch = []
    for key in t_runs:
        for run in t_runs[key]:
            if 'motive2' in tags:
                run.name = 'UB' + str(run.config['random_behv']) + '_' + run.name
            if 'Round/epoch' not in run.summary:
                print('NO EPOCH', run, run.name)
                del t_runs[key]
            else:
                fin_epoch.append(run.summary['Round/epoch'])

    if len(fin_epoch):
        min_epoch = np.min(fin_epoch)
        print(f'finish epochs: {fin_epoch}')

    final_dfs = {}

    for i, key in enumerate(t_runs):
        dfs = {}
        sorted_runs = sorted(t_runs[key], key=lambda x: x.config['sample_seed'])
        for run in sorted_runs:  # t_runs[key]:
            keys = ['_step', 'Round/clock', 'Round/epoch', 'Round/total_updates', 'Round/stale_updates',
                    metric + '/loss',
                    metric + '/acc_top_5', 'Round/compute', 'Round/communicate']

            if 'safa1' in run.tags:
                keys.extend(
                    ['Round/new_compute', 'Round/new_communicate', 'Round/stale_compute', 'Round/stale_communicate'])

            # if 'motive1' in run.tags or 'motive2' in run.tags:
            keys.append('Round/unique_clients')

            if 'adapt' in run.tags:
                keys.append('Round/attended_clients')

            run_hist = run.scan_history(keys)

            df = pd.DataFrame(run_hist)

            df = df.set_index('_step')
            tokens = run.name.split('_')
            seed = run.config['sample_seed']  # int(tokens[-1])
            name = 'part' + str(key) + '_' + '_'.join(tokens[:-1])
            if 'motive2' in tags:
                name = str(run.config['random_behv']) + '_' + name
            if 'safa' in run.tags:
                name = 'safa_' + name
            if 'r-safa' in run.tags:
                name = 'r-safa_' + name
            if 'n-safa' in run.tags:
                name = 'n-safa_' + name
            if method == 'adapt_select' or 'selectadapt' in tags:
                name = name + '_' + 'AS' + str(run.config['adapt_selection'])
                if run.config['stale_update'] != 0 and 'selectadapt' in tags:
                    if 'avail' in tags:
                        name = name.replace('oort', 'abc')
                    else:
                        name = name.replace('random', 'abc')
                    print('new name', name)
            if 'stale' in tags and run.config['stale_update'] != 0:
                name = name.replace('random', 'abc')
                name = name.replace('oort', 'abc')
                print('new name', name)

            if 'scalesystail' in run.tags:
                name = 'SST' + str(run.config['scale_sys']) + '_' + name
            if tags == 'plot_scalesyspercent' and 'scalesyspercent' in run.tags or 'sysadvance' in run.tags:
                if 'scale_sys_percent' in run.config:
                    if run.config['scale_sys_percent'] == 0.0:
                        name = str(0.01) + '_' + str(0.5) + '_' + name
                    else:
                        name = str(run.config['scale_sys_percent'] + 0.01) + '_' + str(
                            run.config['scale_sys']) + '_' + name
                else:
                    # run.config['scale_sys_percent'] = 0.0
                    # run.update()
                    name = 'SST' + str(run.config['scale_sys']) + '_0.0_' + name
                    # name = str(0.01) + '_' + str(0.5) + '_' + name
            if 'scale' in tags and 'scalesyspercent' not in tags:
                name = 'factor' + str(run.config['stale_factor']) + '_' + name
            if name not in dfs:
                dfs[name] = []
            # print(run.name, name, seed, df.size)
            dfs[name].append((seed, run.tags, run.config, df))

        final_dfs[key] = {}
        # process the DFs
        for name in dfs:
            temp_df = None  # pd.DataFrame()
            num_of_runs = len(dfs[name])
            # print(name, num_of_runs)
            # if num_of_runs > 3:
            # print(f'FATAL: {name} experiments has {num_of_runs} seeds, terminating')
            print(f'INFO: Partition {key} experiment {name} has {num_of_runs} seeds {dfs.keys()}')
            # for l in dfs[name]:
            #     seed, config, df = l
            #     if temp_df is None:
            #         temp_df = df
            #     else:
            #         temp_df.add(df, fill_value=0)
            #         #temp_df = pd.DataFrame(temp_df + df)
            #         #temp_df = pd.DataFrame(temp_df.reindex_like(df).fillna(0) + df.fillna(0).fillna(0))
            # temp_df /= 1.0 * len(dfs[name])

            # pd_panel = pd.Panel(dfs[name])
            # temp_df = pd_panel.mean(axis=0)

            temp_dfs = []
            # index = list(range(0,100))
            max = 0
            for v in dfs[name]:
                seed, run_tags, run_config, df = v
                # df = df.reindex(temp_df['Round/epoch'].tolist())
                temp_dfs.append(df)

            temp_df = pd.concat(temp_dfs).groupby(level=0, dropna=False, sort=True).mean()
            # temp_df.reindex(range(0,max))
            if not monotonic(temp_df['Round/epoch'].tolist()):
                print('Non MONTON: ', name, temp_df['Round/epoch'].tolist())
                for df in temp_dfs:
                    print('DFS: ', df['Round/epoch'].tolist())
            # exit(0)
            # plt.figure()
            # temp_df['Test/acc_top_5'].cumsum().plot()
            # plt.show()
            # exit(0)

            print(name, temp_df.size, temp_df['Test/acc_top_5'].mean(), temp_df['Test/acc_top_5'].max(),
                  len(temp_df['Test/acc_top_5'].tolist()))  # , temp_df['Test/acc_top_5'].tolist())
            final_dfs[key][name] = (key, name, run_tags, run_config, temp_df)
    return final_dfs


def plot_dfs(project, exp_type, total_worker, dfs):
    global method, tags, metric, rand_behv
    if clients == 0:
        results_dir = os.path.join(main_dir, 'figs/experiments/', project, tags, metric, method + '_' + str(exp_type),
                                   'total_worker_' + str(total_worker))
    else:
        results_dir = os.path.join(main_dir, 'figs/experiments/', project, tags, metric, method + '_' + str(exp_type),
                                   'total_clients_' + str(clients))

    if sampling_method is not None:
        results_dir = os.path.join(results_dir, 'sampler_' + sampling_method)

    if train_ratio is not None:
        results_dir = os.path.join(results_dir, 'tratio_' + str(train_ratio))

    print('handling:', project, metric, exp_type, total_worker, results_dir)

    for key in dfs:
        temp_dir = os.path.join(results_dir, str(key))
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        fig1, ax1 = plt.subplots()  # figsize=(13, 6))
        fig2, ax2 = plt.subplots()  # figsize=(13, 6))
        fig3, ax3 = plt.subplots()  # figsize=(13, 6))
        if method == 'safa_limitres_no':
            fig3, (ax3, ax33) = plt.subplots(1, 2, sharey=True, facecolor='w')

        if 'motive1' in tags:  # or 'motive2' in tags: # or tags == 'stale_adapt':
            ax3twin = ax3.twinx()

        fig4, ax4 = plt.subplots()  # figsize=(13, 6))
        fig5, ax5 = plt.subplots()  # figsize=(13, 6))
        fig6, ax6 = plt.subplots()  # figsize=(13, 6))

        fig7, ax7 = plt.subplots()  # figsize=(13, 6))
        fig8, ax8 = plt.subplots()  # figsize=(13, 6))
        fig9, ax9 = plt.subplots()  # figsize=(13, 6))

        fig10, ax10 = plt.subplots()  # figsize=(13, 6))
        fig11, ax11 = plt.subplots()  # figsize=(13, 6))
        fig12, ax12 = plt.subplots()  # figsize=(13, 6))

        # dfslist = sorted(final_dfs, key=lambda x: x.run_name, reverse=True)
        # final_dfs = sorted(final_dfs)
        # print('plotting: ',  final_dfs.keys())

        # Ahmed - for SAFA limit the X-axis to resources
        # min_comp_comm = math.inf
        # if method == 'safa_limitres':
        #     for dfs_key in sorted(dfs[key].keys()):
        #         part, run_name, run_tags, run_config, df = dfs[key][dfs_key]
        #         if 'r-safa' in run_tags:
        #             min_comp_comm = (df.iloc[-1]['Round/compute']  + df.iloc[-1]['Round/communicate']) / (60.0 * 60.0)
        #    print('min resources set to: ', min_comp_comm)

        min_acc = math.inf
        min_comp_comm = math.inf
        max_comp_comm = 0
        if method.startswith('safa') or method == 'adapt_select':  # == 'safa_limitres' or method == 'safa':
            for dfs_key in sorted(dfs[key].keys()):
                part, run_name, run_tags, run_config, df = dfs[key][dfs_key]
                acc = df.iloc[-1][metric + '/acc_top_5']
                if method == 'safa_large':
                    if 'r-safa' in run_tags:
                        min_acc = acc
                        min_comp_comm = max_comp_comm = (df.iloc[-1]['Round/compute'] + df.iloc[-1][
                            'Round/communicate']) / (60.0 * 60.0)
                elif min_acc > acc:
                    min_acc = acc
                    min_comp_comm = max_comp_comm = (df.iloc[-1]['Round/compute'] + df.iloc[-1][
                        'Round/communicate']) / (60.0 * 60.0)
                if 'safa' in run_tags:
                    max_comp_comm = (df.iloc[-1]['Round/compute'] + df.iloc[-1]['Round/communicate']) / (60.0 * 60.0)
            print('min acc set to: ', min_acc, 'max comp: ', max_comp_comm, 'min comp: ', min_comp_comm)

        sorted_keys = sorted(dfs[key].keys())
        print('keys:', sorted_keys)
        j = 0
        xypair = []
        for dfs_key in sorted_keys:
            part, run_name, run_tags, run_config, df = dfs[key][dfs_key]

            # staleness = str(run.config['stale_update']) if run.config['stale_update'] >=0 else 'inf'
            # name = run.config['sample_mode'] + '_' + run.config['gradient_policy']
            # if total_worker == 0:
            #     name += '_n' + str(args.config['total_worker'])
            # name = str.lower(name + '_stale_' + staleness)
            if 'plot_motive1' == tags:
                name = str.title(run_config['sample_mode'])
                if run_name.startswith('0'):
                    name += '_' + run_config['gradient_policy']
            elif 'yogi_motive2' == tags:
                # if int(run_config['random_behv']) == -1:
                #     name = 'AllAvail'
                # else:
                #     name = 'DynAvail'
                name = str.title(run_config['sample_mode'])
                if int(run_config['random_behv']) == -1:
                    name += '+All'
                else:
                    name += '+Dyn'
                # if run_config['avail_priority'] == 1:
                #     name += '_Priority'
            elif method.startswith('safa'):
                staleness = str(run_config['stale_update']) if run_config['stale_update'] >= 0 else 'inf'
                name = 'SAFA'
                if 'r-safa' in run_tags:
                    name = 'RELAY'
                elif 'n-safa' in run_tags:
                    # name = str.upper(run_config['sample_mode']) + '_' + str(run_config['total_worker'])
                    name = 'FedAvg_' + str(run_config[
                                               'total_worker'])  # run_config['gradient_policy'] + '_' + str(run_config['total_worker'])
            else:
                if 'scalesystail' in run_tags:
                    if run_config['stale_update'] == 0:
                        name = 'OORT'
                    else:
                        name = 'RELAY'
                    scale_sys = float(run_config['scale_sys'])
                    scale_sys = scale_sys if scale_sys != 0 else 1.0

                    name += '_HS' + str(int(1 / scale_sys))

                elif 'scalesyspercent' in run_tags or 'sysadvance' in run_tags:
                    if run_config['stale_update'] == 0:
                        name = 'Oort'
                    else:
                        name = 'RELAY'
                    scale_sys_percent = int(run_config['scale_sys_percent'] * 100)
                    name += '_HS' + str(percent[scale_sys_percent])

                elif 'scale' in run_tags and 'scale' in tags:
                    name = str(factor[int(run_config['stale_factor'])])

                elif run_config['stale_update'] != 0:
                    if run_config['random_behv'] == -1:
                        if 'stale' in tags:
                            name = 'RELAY'  # name =  str.upper(run_config['sample_mode']) + '_SAA'
                        if 'adapt' in tags:
                            name = 'RELAY'  # str.upper(run_config['sample_mode'])
                            # if run_config['adapt_selection'] == 0:
                            #     name += "+SAA" #+ str.title(config['sample_mode'])
                            # else:
                            #     name += "+SAA+APS"# + str.title(config['sample_mode'])
                            if run_config['adapt_selection'] != 0:
                                name += "+APT"
                        if run_config['stale_update'] > 0:
                            name += '_' + str(run_config['stale_update'])
                    else:
                        if run_config['stale_update'] == 0:
                            name = 'Prioity'
                        else:
                            name = 'RELAY'  # 'RELAY_AS'
                            if run_config['adapt_selection'] != 0 and 'google' in project:
                                name += "+APT"
                elif run_config['avail_priority'] == 1:
                    name = 'Priority'  # 'RELAY_A'
                else:
                    name = str.title(run_config['sample_mode'])

            # name += str(int(round(clock[-1]))) + 'H'
            col_name = str.lower(run_config['sample_mode'] + '_' + run_config['gradient_policy'])

            loss = []
            acc = []
            total_updates = []
            stale_updates = []
            epochs = []
            clock = []
            compute = []
            communicate = []
            total_comp_comm = []

            if 'safa1' in run_tags:
                scompute = []
                scommunicate = []
                ncompute = []
                ncommunicate = []
            # if 'motive1' in run_tags or 'motive2' in run_tags:
            unique = []
            if 'adapt' in run_tags:
                attended = []

            for i, row in df.iterrows():
                epochs.append(row['Round/epoch'])
                clock.append(1.0 * row['Round/clock'] / (60.0 * 60.0))
                if project.startswith('reddit') or project.startswith('stackoverflow'):
                    loss.append(row[metric + '/loss'] ** 2)
                else:
                    loss.append(row[metric + '/loss'])
                acc.append(row[metric + '/acc_top_5'])
                total_updates.append(row['Round/total_updates'])
                stale_updates.append(row['Round/stale_updates'])
                compute.append(row['Round/compute'] / (60.0 * 60.0))
                communicate.append(row['Round/communicate'] / (60.0 * 60.0))
                total_comp_comm.append(compute[-1] + communicate[-1])

                if 'safa1' in run_tags:
                    ncompute.append(row['Round/new_compute'] / (60.0 * 60.0))
                    ncommunicate.append(row['Round/new_communicate'] / (60.0 * 60.0))
                    scompute.append(row['Round/stale_compute'] / (60.0 * 60.0))
                    scommunicate.append(row['Round/stale_communicate'] / (60.0 * 60.0))
                # if 'motive1' in run_tags or 'motive2' in run_tags:
                if key == 0:
                    unique.append(100.0 * row['Round/unique_clients'] / 2084)
                else:
                    unique.append(100.0 * row['Round/unique_clients'] / 3000)

                if 'adapt' in run_tags:
                    attended.append(row['Round/attended_clients'])

                if method.startswith('safa') and acc[-1] >= min_acc * 0.99:
                    print(f'BREAKING at {epochs[-1]} of accuracy {acc[-1]}')
                    break

                if tags == 'stale_selectadapt' and run_config['adapt_selection'] == 0 and run_config[
                    'stale_update'] == -1 and epochs[-1] >= 530:
                    break

            if len(epochs) <= 0:
                print('EMPTY epochs', run_name)
                continue

            marker = None  # markers['stale_update_' +  str(run_config['stale_update'])]
            linestyle = '-'  # linestyles[str(run_config['gradient_policy'])]
            if 'fig1' in tags and run_config['total_worker'] == 10:
                linestyle = '--'

            color = None
            if name in colors:
                color = colors[name]

            sacc = smooth(acc, kernel_size)
            sloss = smooth(loss, kernel_size)
            supdates = total_updates  # smooth(total_updates, kernel_size)

            ax1.plot(epochs[:len(sacc)], sacc, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax2.plot(clock[:len(sacc)], sacc, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax4.plot(epochs[1:len(sloss)], sloss[1:], linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax5.plot(clock[1:len(sloss)], sloss[1:], linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax7.plot(epochs[:len(supdates)], supdates, linestyle=linestyle, marker=marker, ms=5, label=name,
                     color=color)
            ax8.plot(clock[:len(supdates)], supdates, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)

            ax10.plot(epochs[:len(unique)], unique, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax11.plot(clock[:len(unique)], unique, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)

            line, = ax3.plot(total_comp_comm[:len(sacc)], sacc, linestyle=linestyle, marker=marker, ms=5, label=name,
                             color=color)
            thresh = 0.03
            x = total_comp_comm[len(sacc) - 1] * 1
            y = sacc[-1] * 1.02

            if 'avail_yogi' == tags:
                x *= 0.95

            for lx, ly in xypair:
                x_ratio = (x - lx) / lx
                y_ratio = (y - ly) / ly
                print('y: ', y_ratio, 'x: ', x_ratio, thresh)
                if abs(x_ratio) < thresh:
                    if abs(y_ratio) < thresh * 2:
                        print('CHANGE y:', y_ratio, thresh)
                        y -= y * 0.1
                    # print('x:', x_ratio, thresh)
                    # if y > ly:
                    #     y += y * 0.05  # min(1.03, 1 + y_ratio)
                    # else:
                    #     y -= y * 0.05

                    # if abs(y_ratio) < thresh:
                    #     #y -=  y * 0.07 #min(0.97, 1 - x_ratio)
                    #     if y > ly:
                    #         y += y * 0.05 #min(1.03, 1 + y_ratio)
                    #     else:
                    #         y -= y * 0.03
                    # else:
                    #     y +=  y * 0.05 #min(1.03, 1 + y_ratio)
                    # x_ratio = 1.0 * abs(x - lx) / lx
                elif abs(y_ratio) < thresh:
                    if abs(x_ratio) < thresh * 2:
                        print('CHANGE y:', y_ratio, thresh)
                        y -= y * 0.1

            if name == 'FedAvg_100':
                x = total_comp_comm[len(sacc) - 1] * 1.5
            elif name == 'FedAvg_10':
                x = total_comp_comm[len(sacc) - 1] * 0.55
            if 'fig1' in tags:
                y = sacc[-1] * 1.05
            if 'scalesyspercent' in tags:
                if sampling_method == 'oort' and key == 1 and run_config['scale_sys_percent'] != 0:
                    y = sacc[-1] * 1.1
                if sampling_method == 'relay':
                    if key == -1 and run_config['scale_sys_percent'] == 0.75:
                        print('0.7555555555')
                        x = total_comp_comm[len(sacc) - 1] * 1.05
                    elif key == 1 and run_config['scale_sys_percent'] == 0.25:
                        y = sacc[-1] * 1.0
            if 'motive' in tags:
                if key == 0:
                    x *= 0.92
                else:
                    x *= 0.97
            if 'motive' in tags and 'oort' in dfs_key:

                x *= 0.95
                if 'UB1' in dfs_key and key == 0:
                    y -= y * 0.05

            if 'fig1' in tags:
                x *= 0.9
                if name == 'FedAvg_100':
                    y *= 0.97
                ax3.text(x, y, "{:.1f}".format(float(clock[-1])) + 'H', fontsize=FONT_SIZE, weight='bold',
                         color=line.get_color())
            else:
                ax3.text(x, y, str(int(round(clock[-1]))) + 'H', fontsize=FONT_SIZE, weight='bold',
                         color=line.get_color())
            if tags != 'stale_selectadapt':
                xypair.append((x, y))
            print(name, total_comp_comm[len(sacc) - 1], x, y, xypair)

            line, = ax6.plot(total_comp_comm[1:len(sloss)], sloss[1:], linestyle=linestyle, marker=marker, ms=5,
                             label=name, color=color)
            ax6.text(total_comp_comm[len(sloss) - 1], sloss[-1], str(int(round(clock[-1]))) + 'H', weight='bold',
                     fontsize=FONT_SIZE, color=line.get_color())

            ax9.plot(total_comp_comm[:len(supdates)], supdates, linestyle=linestyle, marker=marker, ms=5, label=name,
                     color=color)
            ax12.plot(total_comp_comm[:len(unique)], unique, linestyle=linestyle, marker=marker, ms=5, label=name,
                      color=color)

            if 'safa1' in run_tags:
                total_comp_comm = [ncompute[i] + ncommunicate[i] + scompute[i] + scommunicate[i] for i in
                                   range(0, len(ncompute))]
                print('resource1: ', total_comp_comm)
                line, = ax3.plot(total_comp_comm[:len(sacc)], sacc, linestyle='--', marker=marker, ms=5,
                                 label=name + '+O', color=color)
                if 'fig1' in run_tags:
                    ax3.text(total_comp_comm[len(sacc) - 1] * 0.8, sacc[-1] * 1.05,
                             "{:.1f}".format(float(clock[-1])) + 'H', fontsize=FONT_SIZE, weight='bold',
                             color=line.get_color())
                else:
                    ax3.text(total_comp_comm[len(sacc) - 1] * 0.97, sacc[-1] * 1.01, str(int(round(clock[-1]))) + 'H',
                             fontsize=FONT_SIZE, weight='bold')
                ax6.plot(total_comp_comm[:len(sloss)], sloss, linestyle=':', marker=marker, ms=5, label=name + '+O',
                         color=color)
                ax9.plot(total_comp_comm[:len(supdates)], supdates, linestyle=":", marker=marker, ms=5,
                         label=name + '+O', color=color)

            if 'motive1' in tags:  # or 'motive2' in run_tags:
                line, = ax3twin.plot(total_comp_comm[:len(unique)], unique, linestyle=':', color=color)  # mcolors[j])

            j += 1
        ##############
        #### Legend
        handles, labels = ax3.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        if tags == 'plot_scalesyspercent':
            labels = [x.split('_')[1] for x in labels]
        # Replace the name of the scheme
        new_labels = []
        for i in range(len(labels)):
            if 'RELAY' in labels[i]:
                new_labels.append(labels[i].replace('RELAY', 'REFL'))
            else:
                new_labels.append(labels[i])
        labels = new_labels
        print(f'Number of labels {len(labels)}')
        if len(labels) < 3:  # or 'motive2' in tags:
            title = None

            ax1.legend(handles, labels, title=title, ncol=2, loc='best')
            ax2.legend(handles, labels, title=title, ncol=2, loc='best')
            if 'safa1' in run_tags or 'motive1' in run_tags or 'motive2' in run_tags:
                ax3.legend(handles, labels, title=title, ncol=2, loc='lower right')
            else:
                ax3.legend(handles, labels, title=title, ncol=2, loc='best')

            ax4.legend(handles, labels, title=title, ncol=2, loc='best')
            ax5.legend(handles, labels, title=title, ncol=2, loc='best')
            ax6.legend(handles, labels, title=title, ncol=2, loc='best')
            ax7.legend(handles, labels, title=title, ncol=2, loc='best')
            ax8.legend(handles, labels, title=title, ncol=2, loc='best')
            ax9.legend(handles, labels, title=title, ncol=2, loc='best')
        else:
            figLegend = plt.figure(figsize=(2, 0.1))
            # # produce a legend for the objects in the other figure
            plt.figlegend(handles, labels, loc='center', ncol=7)  # ,title='Compression Ratio')
            figLegend.savefig(temp_dir + '/legend.pdf', bbox_inches='tight')
            plt.close(figLegend)
            exit(0)
        ##############
        ax1.set_ylabel(metric + ' Accuracy', fontsize=FONT_SIZE)
        ax1.set_xlabel('Training Rounds', fontsize=FONT_SIZE)
        ax1.grid(True)
        fig1.set_tight_layout(True)
        fig1.savefig(temp_dir + "/acc_round.pdf", bbox_inches='tight')

        #############
        ax2.set_ylabel(metric + 'Accuracy', fontsize=FONT_SIZE)
        ax2.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        ax2.grid(True)
        fig2.set_tight_layout(True)
        fig2.savefig(temp_dir + "/acc_time.pdf", bbox_inches='tight')

        #############
        ax3.set_ylabel(metric + ' Accuracy (%)', fontsize=FONT_SIZE)
        if 'motive1' in tags:  # tags == 'plot_motive1' or tags == 'plot_motive2':
            ax3twin.set_ylabel('Unique Participants (%)', fontsize=FONT_SIZE)
            ax3twin.set_ylim(0, 100)
        if 'motive' in tags:  # tags == 'plot_motive1' or tags == 'plot_motive2':
            if key == 0:
                ax3.set_ylim(10, 80)
            else:
                ax3.set_ylim(10, 30)
        if tags == 'fig1':  # or tags == 'stale_selectadapt':
            ax3.set_xscale('log')
            ax3.set_ylim(10, 55)
            ax3.set_xlabel('Cumulative resource usage (hours) - log', fontsize=FONT_SIZE)
        else:
            ax3.set_xlabel('Cumulative resource usage (hours)', fontsize=FONT_SIZE)
        # ax2.legend(title='Methods', ncol=2)
        ax3.grid(True)
        if tags == 'plot_scalesyspercent' or 'sysadvance' in run_tags:
            ax3.set_xlim(0, 1000)
            if key != 0 and key != -1:
                ax3.set_ylim(10, 50)
            else:
                ax3.set_ylim(10, 85)
        if 'selectadapt' in tags:
            if key != 0 and key != -1:
                ax3.set_ylim(10, 60)
        # else:
        #     ax3.set_ylim(0, 80)
        fig3.set_tight_layout(True)
        fig3.savefig(temp_dir + "/acc_com.pdf", bbox_inches='tight')

        #############
        if project.startswith('reddit') or project.startswith('stackoverflow'):
            ax4.set_ylabel(metric + ' Preplexity', fontsize=FONT_SIZE)
        else:
            ax4.set_ylabel(metric + ' Loss', fontsize=FONT_SIZE)
        ax4.set_xlabel('Training Rounds', fontsize=FONT_SIZE)
        # ax3.legend(title='Methods', ncol=2)
        ax4.grid(True)
        fig4.set_tight_layout(True)
        fig4.savefig(temp_dir + "/loss_round.pdf", bbox_inches='tight')

        #############
        if project.startswith('reddit') or project.startswith('stackoverflow'):
            ax5.set_ylabel(metric + ' Preplexity', fontsize=FONT_SIZE)
        else:
            ax5.set_ylabel(metric + ' Loss', fontsize=FONT_SIZE)
        ax5.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        # ax4.legend(title='Methods', ncol=2)
        ax5.grid(True)
        fig5.set_tight_layout(True)
        fig5.savefig(temp_dir + "/loss_time.pdf", bbox_inches='tight')

        #############
        if project.startswith('reddit') or project.startswith('stackoverflow'):
            ax6.set_ylabel('Test Preplexity', fontsize=FONT_SIZE)
        else:
            ax6.set_ylabel('Test Loss', fontsize=FONT_SIZE)
        ax6.set_xlabel('Cumulative resource usage (hours)', fontsize=FONT_SIZE)
        # ax2.legend(title='Methods', ncol=2)
        ax6.grid(True)
        fig6.set_tight_layout(True)
        fig6.savefig(temp_dir + "/loss_com.pdf", bbox_inches='tight')

        #############
        ax7.set_ylabel('Total Updates', fontsize=FONT_SIZE)
        ax7.set_xlabel('Training Rounds', fontsize=FONT_SIZE)
        # ax5.legend(title='Methods', ncol=2)
        ax7.grid(True)
        ax7.set_ylim(7, 13)
        fig7.set_tight_layout(True)
        fig7.savefig(temp_dir + "/total_updates_round.pdf", bbox_inches='tight')

        #############
        ax8.set_ylabel('Total Updatess', fontsize=FONT_SIZE)
        ax8.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        # ax6.legend(title='Methods', ncol=2)
        ax8.grid(True)
        fig8.set_tight_layout(True)
        fig8.savefig(temp_dir + "/total_updates_time.pdf", bbox_inches='tight')

        #############
        ax9.set_ylabel('Total Updatess', fontsize=FONT_SIZE)
        ax9.set_xlabel('Cumulative resource usage (hours)', fontsize=FONT_SIZE)
        # ax6.legend(title='Methods', ncol=2)
        ax9.grid(True)
        fig9.set_tight_layout(True)
        fig9.savefig(temp_dir + "/total_updates_com.pdf", bbox_inches='tight')

        #############
        ax10.set_ylabel('% of Unique Learners', fontsize=FONT_SIZE)
        ax10.set_xlabel('Training Rounds', fontsize=FONT_SIZE)
        # ax6.legend(title='Methods', ncol=2)
        ax10.grid(True)
        ax10.set_ylim(0, 100)
        fig10.set_tight_layout(True)
        fig10.savefig(temp_dir + "/unqiue_learners_rounds.pdf", bbox_inches='tight')

        #############
        ax11.set_ylabel('% of Unique Learners', fontsize=FONT_SIZE)
        ax11.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        # ax6.legend(title='Methods', ncol=2)
        ax11.grid(True)
        ax11.set_ylim(0, 100)
        fig11.set_tight_layout(True)
        fig11.savefig(temp_dir + "/unqiue_learners_time.pdf", bbox_inches='tight')

        #############
        ax12.set_ylabel('% of Unique Learners', fontsize=FONT_SIZE)
        ax12.set_xlabel('Cumulative resource usage (hours)', fontsize=FONT_SIZE)
        # ax6.legend(title='Methods', ncol=2)
        ax12.grid(True)
        ax12.set_ylim(0, 100)
        fig12.set_tight_layout(True)
        fig12.savefig(temp_dir + "/unqiue_learners_com.pdf", bbox_inches='tight')

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
        plt.close(fig6)
        plt.close(fig7)
        plt.close(fig8)
        plt.close(fig9)
        plt.close(fig10)
        plt.close(fig11)


def plot_exps_df(project, exp_type, worker):
    global method
    dfs = get_dfs(project, exp_type, worker)
    print('DFS: ', len(dfs), dfs.keys())
    plot_dfs(project, exp_type, worker, dfs)


def plot_safa(project, metric='Test'):
    results_dir = os.path.join(main_dir, 'figs/experiments/', tags, metric)
    print('handling:', project, metric, results_dir)

    runs = api.runs('refl/' + project)
    check_tags = []
    if tags != '':
        check_tags.extend(str(tags).split('_'))

    print('inputs: ', project, exp_type, worker, check_tags)
    t_runs = {}
    for run in runs:
        if 'hidden' in run.tags or 'wrong' in run.tags or 'plot' not in run.tags:
            continue
        if run.state == 'running' or run.state == 'killed':
            continue
        # Ahmed - exclude infinity for stale updates
        if 'stale_update' in run.config and run.config['stale_update'] == -1:
            continue
        if 'safa' in run.tags or 'r-safa' in run.tags or 'n-safa' in run.tags:
            if run.config['partitioning'] not in t_runs:
                t_runs[run.config['partitioning']] = []
            t_runs[run.config['partitioning']].append(run)
            print('Add: ', run.tags, run, run.name)

    fin_update = {}
    for key in t_runs:
        if key not in fin_update:
            fin_update[key] = []
        for run in t_runs[key]:
            temp = []
            run_hist = run.scan_history(keys=['Round/total_updates', metric + '/loss'])
            for i, row in enumerate(run_hist):
                temp.append(row['Round/total_updates'])
            print(run.tags, run.name, sum(temp), temp)
            fin_update[key].append(sum(temp))

    for i, key in enumerate(t_runs):
        temp_dir = os.path.join(results_dir, str(key))
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        fig1, ax1 = plt.subplots()  # figsize=(13, 6))
        fig2, ax2 = plt.subplots()  # figsize=(13, 6))
        fig3, ax3 = plt.subplots()  # figsize=(13, 6))

        fig4, ax4 = plt.subplots()  # figsize=(13, 6))
        fig5, ax5 = plt.subplots()  # figsize=(13, 6))
        fig6, ax6 = plt.subplots()  # figsize=(13, 6))

        fig7, ax7 = plt.subplots()  # figsize=(13, 6))
        fig8, ax8 = plt.subplots()  # figsize=(13, 6))
        fig9, ax9 = plt.subplots()  # figsize=(13, 6))

        min_updates = np.min(fin_update[key]) if len(fin_update[key]) else math.inf
        print(key, ' : ', min_updates, ' : ', fin_update[key])

        runlist = sorted(t_runs[key], key=lambda x: x.name)  # , reverse=True)
        for run in runlist:

            staleness = str(run.config['stale_update']) if run.config['stale_update'] >= 0 else 'inf'
            name = 'SAFA' if str(run.tags[-1]).startswith('safa') else 'RELAY'
            name = name
            print(key, name, run, run.tags, run.name)  # , run.config)

            loss = []
            acc = []
            total_updates = []
            stale_updates = []
            epochs = []
            clock = []
            compute = []
            communicate = []

            keys = ['Round/clock', 'Round/epoch', 'Round/total_updates', 'Round/stale_updates', metric + '/loss',
                    metric + '/acc_top_5', 'Round/compute', 'Round/communicate']

            if 'safa1' in run.tags:
                scompute = []
                scommunicate = []
                ncompute = []
                ncommunicate = []
                keys.extend(
                    ['Round/new_compute', 'Round/new_communicate', 'Round/stale_compute', 'Round/stale_communicate'])

            run_hist = run.scan_history(keys)
            for i, row in enumerate(run_hist):
                if len(total_updates) > 0 and sum(total_updates) > min_updates:
                    print(run.name, ' : ', sum(total_updates), ' : ', min_updates)
                    break
                epochs.append(row['Round/epoch'])
                clock.append(row['Round/clock'] / (60.0 * 60.0))
                if project.startswith('reddit') or project.startswith('stackoverflow'):
                    loss.append(row[metric + '/loss'] ** 2)
                else:
                    loss.append(row[metric + '/loss'])
                acc.append(row[metric + '/acc_top_5'])
                total_updates.append(row['Round/total_updates'])
                stale_updates.append(row['Round/stale_updates'])
                compute.append(row['Round/compute'] / (60.0 * 60.0))
                communicate.append(row['Round/communicate'] / (60.0 * 60.0))

                if 'safa1' in run.tags:
                    ncompute.append(row['Round/new_compute'] / (60.0 * 60.0))
                    ncommunicate.append(row['Round/new_communicate'] / (60.0 * 60.0))
                    scompute.append(row['Round/stale_compute'] / (60.0 * 60.0))
                    scommunicate.append(row['Round/stale_communicate'] / (60.0 * 60.0))

            print(run.name, ' : ', sum(total_updates), ' : ', min_updates, total_updates)

            # fmt = fmts[col_name]
            marker = markers['stale_update_' + str(run.config['stale_update'])]
            linestyle = linestyles[str(run.config['gradient_policy'])]

            color = None  # colors[name]

            total_comp_comm = [compute[i] + communicate[i] for i in range(0, len(compute))]
            print('resource: ', total_comp_comm)

            sacc = smooth(acc, kernel_size)
            sloss = smooth(loss, kernel_size)
            supdates = smooth(total_updates, kernel_size)

            ax1.plot(epochs[:len(sacc)], sacc, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax2.plot(clock[:len(sacc)], sacc, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax3.plot(total_comp_comm[:len(sacc)], sacc, linestyle=linestyle, marker=marker, ms=5, label=name,
                     color=color)
            ax3.text(total_comp_comm[len(sacc) - 1], sacc[-1], str(round(clock[-1])) + ' hours', fontsize=FONT_SIZE / 2)
            # ax3.text(total_comp_comm[-1] - 1.0 * total_comp_comm[-1]/8, test_acc[-1], 'updates=' + str(sum(total_updates)), bbox=dict(facecolor='red', alpha=0.5), fontsize=FONT_SIZE/2)
            # ax3.text(total_comp_comm[-1] - 1.0 * total_comp_comm[-1]/8, test_acc[-1] - 2, 'stale=' + str(sum(stale_updates)), bbox=dict(facecolor='red', alpha=0.5), fontsize=FONT_SIZE/2)

            ax4.plot(epochs[:len(sloss)], sloss, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax5.plot(clock[:len(sloss)], sloss, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax6.plot(total_comp_comm[:len(sloss)], sloss, linestyle=linestyle, marker=marker, ms=5, label=name,
                     color=color)
            # ax6.text(total_comp_comm[-1] - 1.0 * total_comp_comm[-1]/8, test_loss[-1], 'updates=' + str(sum(total_updates)), bbox=dict(facecolor='red', alpha=0.5), fontsize=FONT_SIZE/2)
            # ax6.text(total_comp_comm[-1] - 1.0 * total_comp_comm[-1]/8, test_loss[-1] - 2, 'stale=' + str(sum(stale_updates)), bbox=dict(facecolor='red', alpha=0.5), fontsize=FONT_SIZE/2)

            ax7.plot(epochs[:len(supdates)], supdates, linestyle=linestyle, marker=marker, ms=5, label=name,
                     color=color)
            ax8.plot(clock[:len(supdates)], supdates, linestyle=linestyle, marker=marker, ms=5, label=name, color=color)
            ax9.plot(total_comp_comm[:len(supdates)], supdates, linestyle=linestyle, marker=marker, ms=5, label=name,
                     color=color)

            if 'safa1' in run.tags:
                total_comp_comm = [ncompute[i] + ncommunicate[i] + scompute[i] + scommunicate[i] for i in
                                   range(0, len(ncompute))]
                print('resource1: ', total_comp_comm)
                ax3.plot(total_comp_comm[:len(sacc)], sacc, linestyle=linestyle, marker=marker, ms=5, label=name + '+O',
                         color=color)
                ax3.text(total_comp_comm[len(sacc) - 1], sacc[-1], str(round(clock[-1])) + ' hours',
                         fontsize=FONT_SIZE / 2)
                ax6.plot(total_comp_comm[:len(sloss)], sloss, linestyle=linestyle, marker=marker, ms=5,
                         label=name + '+O', color=color)
                ax9.plot(total_comp_comm[:len(supdates)], supdates, linestyle=linestyle, marker=marker, ms=5,
                         label=name + '+O', color=color)

        ##############
        #### Legend
        handles, labels = ax1.get_legend_handles_labels()
        if len(labels) < 3:
            ax1.legend(title='Methods', ncol=3, loc='best')
            ax2.legend(title='Methods', ncol=3, loc='best')
            ax3.legend(title='Methods', ncol=3, loc='best')
            ax4.legend(title='Methods', ncol=3, loc='best')
            ax5.legend(title='Methods', ncol=3, loc='best')
            ax6.legend(title='Methods', ncol=3, loc='best')
            ax7.legend(title='Methods', ncol=3, loc='best')
            ax8.legend(title='Methods', ncol=3, loc='best')
            ax9.legend(title='Methods', ncol=3, loc='best')
        else:
            figLegend = plt.figure(figsize=(2, 0.1))
            # # produce a legend for the objects in the other figure
            plt.figlegend(handles, labels, loc='center', ncol=6)  # ,title='Compression Ratio')
            figLegend.savefig(temp_dir + '/legend.pdf', bbox_inches='tight')
            plt.close(figLegend)

        ##############
        ax1.set_ylabel(metric + ' Accuracy', fontsize=FONT_SIZE)
        ax1.set_xlabel('Training Rounds', fontsize=FONT_SIZE)
        # ax1.legend(title='Methods', ncol=2)
        # ax1.legend(handles2, labels2, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #               ncol=5, mode="expand", borderaxespad=0.)
        # ax1.set_ylim([0,0.5])
        ax1.grid(True)
        # ax1.set_yscale('log')
        fig1.set_tight_layout(True)
        fig1.savefig(temp_dir + "/acc_round.pdf", bbox_inches='tight')

        #############
        ax2.set_ylabel(metric + ' Accuracy', fontsize=FONT_SIZE)
        ax2.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        # ax2.legend(title='Methods', ncol=2)
        ax2.grid(True)
        fig2.set_tight_layout(True)
        fig2.savefig(temp_dir + "/acc_time.pdf", bbox_inches='tight')

        #############
        ax3.set_ylabel(metric + ' Accuracy', fontsize=FONT_SIZE)
        ax3.set_xlabel('Cumulative resource usage (hours)', fontsize=FONT_SIZE)
        # ax2.legend(title='Methods', ncol=2)
        ax3.grid(True)
        fig3.set_tight_layout(True)
        fig3.savefig(temp_dir + "/acc_com.pdf", bbox_inches='tight')

        #############
        if project.startswith('reddit') or project.startswith('stackoverflow'):
            ax4.set_ylabel(metric + ' Preplexity', fontsize=FONT_SIZE)
        else:
            ax4.set_ylabel(metric + ' Loss', fontsize=FONT_SIZE)
        ax4.set_xlabel('Training Rounds', fontsize=FONT_SIZE)
        # ax3.legend(title='Methods', ncol=2)
        ax4.grid(True)
        fig4.set_tight_layout(True)
        fig4.savefig(temp_dir + "/loss_round.pdf", bbox_inches='tight')

        #############
        if project.startswith('reddit') or project.startswith('stackoverflow'):
            ax5.set_ylabel(metric + ' Preplexity', fontsize=FONT_SIZE)
        else:
            ax5.set_ylabel(metric + ' Loss', fontsize=FONT_SIZE)
        ax5.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        # ax4.legend(title='Methods', ncol=2)
        ax5.grid(True)
        fig5.set_tight_layout(True)
        fig5.savefig(temp_dir + "/loss_time.pdf", bbox_inches='tight')

        #############
        if project.startswith('reddit') or project.startswith('stackoverflow'):
            ax6.set_ylabel(metric + ' Preplexity', fontsize=FONT_SIZE)
        else:
            ax6.set_ylabel(metric + ' Loss', fontsize=FONT_SIZE)
        ax6.set_xlabel('Cumulative resource usage (hours)', fontsize=FONT_SIZE)
        # ax2.legend(title='Methods', ncol=2)
        ax6.grid(True)
        fig6.set_tight_layout(True)
        fig6.savefig(temp_dir + "/loss_com.pdf", bbox_inches='tight')

        #############
        ax7.set_ylabel('Total Updates', fontsize=FONT_SIZE)
        ax7.set_xlabel('Training Rounds', fontsize=FONT_SIZE)
        # ax5.legend(title='Methods', ncol=2)
        ax7.grid(True)
        fig7.set_tight_layout(True)
        fig7.savefig(temp_dir + "/total_updates_round.pdf", bbox_inches='tight')

        #############
        ax8.set_ylabel('Total Updatess', fontsize=FONT_SIZE)
        ax8.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        # ax6.legend(title='Methods', ncol=2)
        ax8.grid(True)
        fig8.set_tight_layout(True)
        fig8.savefig(temp_dir + "/total_updates_time.pdf", bbox_inches='tight')

        #############
        ax9.set_ylabel('Total Updatess', fontsize=FONT_SIZE)
        ax9.set_xlabel('Cumulative resource usage (hours)', fontsize=FONT_SIZE)
        # ax6.legend(title='Methods', ncol=2)
        ax9.grid(True)
        fig9.set_tight_layout(True)
        fig9.savefig(temp_dir + "/total_updates_com.pdf", bbox_inches='tight')

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
        plt.close(fig6)
        plt.close(fig7)
        plt.close(fig8)
        plt.close(fig9)


def get_unfinished():
    global tags, metric, rand_behv, project, worker, exp_type
    runs = api.runs('refl/' + project)

    check_tags = []
    if tags != '':
        check_tags.extend(str(tags).split('_'))

    print('inputs: ', project, exp_type, worker, check_tags)
    for run in runs:
        if 'hidden' in run.tags or 'wrong' in run.tags:
            continue
        if len(check_tags) > 0:
            if len(run.tags) <= 0:  # or len(run.tags) != len(check_tags):
                continue
            else:
                all_ok = True
                for tag in check_tags:
                    if tag not in run.tags:
                        all_ok = False
                        break
                if not all_ok:
                    continue
        if run.state == 'running' or run.state == 'killed':
            continue
        if 'exp_type' in run.config and run.config['exp_type'] != exp_type:
            continue
        if 'rand_behv' in run.config and run.config['rand_behv'] != rand_behv:
            continue
        if worker != 0 and 'total_worker' in run.config and run.config['total_worker'] != worker:
            continue
        if 'Round/epoch' in run.summary and run.summary['Round/epoch'] < max_epoch:
            print('REDO: ', run.summary['Round/epoch'], run.config['partitioning'], run.url)


print(sys.argv)
restrict = False
method = sys.argv[1]
project = sys.argv[2] if len(sys.argv) > 2 else ''
tags = sys.argv[3] if len(sys.argv) > 3 else None
# if tags == 'avail_yogi' or tags == 'avail_stale':
#      restrict = True
metric = sys.argv[4] if len(sys.argv) > 4 else 'Test'
exp_type = int(sys.argv[5]) if len(sys.argv) > 5 else None
worker = int(sys.argv[6]) if len(sys.argv) > 6 else 0
clients = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rand_behv = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7].strip() != "None" else None
sampling_method = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8].strip() != "None" else None
train_ratio = sys.argv[9] if len(sys.argv) > 9 and sys.argv[9].strip() != "None" else None
print(method, project, tags, metric, exp_type, worker, clients, rand_behv, sampling_method, train_ratio)

if method == 'exp_type':
    if project == '' and exp_type == '' and worker == '':
        for project in proj_names:
            for exp_type in exp_types:
                for worker in total_workers:
                    plot_exps_df(project, int(exp_type), int(worker))
    elif exp_type == '' and exp_type == '':
        for exp_type in exp_types:
            for worker in total_workers:
                plot_exps_df(project, int(exp_type), int(worker))
    elif worker == '':
        for worker in total_workers:
            plot_exps_df(project, int(exp_type), int(worker))

    else:
        plot_exps_df(project, int(exp_type), int(worker))
else:
    plot_exps_df(project, int(exp_type), int(worker))

if method == 'unfinished':
    get_unfinished()