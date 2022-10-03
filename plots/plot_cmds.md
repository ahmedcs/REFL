#Commands to produce the plots in the paper 

The commands below are based on the results generated in our experiments and is specific to the tag naming we used. Different tag names require adjustment of the commands

### Commands to plot the figures of the motivation part
```
Fig2: python plots/plot_exp.py 'safa1' google_speech_resnet34 'fig1' 'Test' 0 1000
Fig3: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'plot_motive1' 'Test' 1 10
Fig4: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'plot_motive2' 'Test' 1 10
```

### Commands to plot the main figures for Google Speech Benchmark
```
Selection Methods: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'avail_yogi' 'Test' 1 10
Staleness Aggregation: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'stale_yogi' 'Test' 1 10
Compare with Safa - Fig 10: python plots/plot_exp.py 'safa' google_speech_resnet34 'fig2' 'Test' 0 1000
Selection Adaptation (APT) - Fig 11: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'avail_selectadapt' 'Test' 0 50
```
### Commands to plot figures of the future proof
```
Oort-Future: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'plot_scalesyspercent' 'Test' 1 10 1 oort
REFL-Future: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'plot_scalesyspercent' 'Test' 1 10 1 relay
Safa-large: python plots/plot_exp.py 'safa_large' google_speech_resnet34 'fig3' 'Test' 0 3000
```

### Command to plot the figures related to the stale updates mainiplation methods of the SAA module
```
stale update methods: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'avail_yogi_scale' 'Test' 1 10
```
### Command to plot the convergence plots for extended number of rounds
```
avail_yogi_converge - Fig 9: python plots/plot_exp.py 'exp_type' google_speech_resnet34 'avail_yogi_converge' 'Test' 1 10
```

#### Commands to plot the figures of the other benchmarks
```
Reddit:  python plots/plot_exp.py 'exp_type' reddit_albert-base-v2 'plot_avail' 'Test' 1 10
Stackoverflow: python plots/plot_exp.py 'exp_type' stackoverflow_albert-base-v2 'plot_avail' 'Test' 1 10
openimg:  python plots/plot_exp.py 'exp_type' openimg_shufflenet 'plot_avail' 'Test' 1 10 None None 1.0
cifar10: python plots/plot_exp.py 'exp_type' cifar10_resnet18 'fedavg_avail' 'Test' 1 10
```
