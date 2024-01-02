# Risk-Aware Deep Reinforcement Learning for Robot Crowd Navigation
This repository contains the codes for our paper titled "Risk-Aware Deep Reinforcement Learning for Robot Crowd Navigation". 
For more details, please refer to the [published paper](https://doi.org/10.3390/electronics12234744).

## Abstract
Ensuring safe and efficient navigation in crowded environments is a critical goal for assistive robots. Recent studies have emphasized the potential of deep reinforcement learning techniques to enhance robots’ navigation capabilities in the presence of crowds. However, current deep reinforcement learning methods often face the challenge of robots freezing as crowd density increases. To address this issue, a novel risk-aware deep reinforcement learning approach is proposed in this paper. The proposed method integrates a risk function to assess the probability of collision between the robot and pedestrians, enabling the robot to proactively prioritize pedestrians with a higher risk of collision. Furthermore, the model dynamically adjusts the fusion strategy of learning-based and risk-aware-based features, thereby improving the robustness of robot navigation. Evaluations were conducted to determine the effectiveness of the proposed method in both low- and high-crowd density settings. The results exhibited remarkable navigation success rates of 98.0% and 93.2% in environments with 10 and 20 pedestrians, respectively. These findings emphasize the robust performance of the proposed method in successfully navigating through crowded spaces. Additionally, the approach achieves navigation times comparable to those of state-of-the-art methods, confirming its efficiency in accomplishing navigation tasks. The generalization capability of the method was also rigorously assessed by subjecting it to testing in crowd environments exceeding the training density. Notably, the proposed method attains an impressive navigation success rate of 90.0% in 25-person environments, surpassing the performance of existing approaches and establishing itself as a state-of-the-art solution. This result highlights the versatility and effectiveness of the proposed method in adapting to various crowd densities and further reinforces its applicability in real-world scenarios.


The following figure Illustration of the proposed robot–human interaction method. The proposed method integrates learning-based and risk-aware strategy-based robot–human interaction features. The gray arrow represents the velocity of the agent.
<div align="robot–human interaction">
<img src=./figures/figure1.png width=50% />
</div>

The network architecture is shown in figure 2. The size of the scene is 12 m × 12 m and the environment contains 20 pedestrians and one mobile robot with the visualization radius of 5 m . The neural network consists of the spatial feature encoder, temporal feature encoder and Figure 2. The network architecture of our proposed approach. The size of the scene is 12 m × 12 m and the environment contains 20 pedestrians and one mobile robot with the visualization radius of 5 m. The neural network consists of the spatial feature encoder, temporal feature encoder and policy-learning subnetwork. The spatial feature encoder integrates learning-based and risk-aware-based  features, which leads to robust feature expression for policy learning.
<div align="architecture">
<img src=./figures/figure2.png width=50% />
</div>


## Setup
1. In a conda environment or virtual environment with Python 3.x, install the required python package
```
pip install -r requirements.txt
```
* All dependencies can be found in [requirements_all.txt](./requirements_all.txt)

2. Install Pytorch 2.0.1 following the instructions [here](https://pytorch.org/get-started/previous-versions/)

3. Install [OpenAI Baselines](https://github.com/openai/baselines#installation) 
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

4. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library



## Run the code
### Training
- Modify the configurations.
  1. Environment configurations: Modify `crowd_nav/configs/config.py`. Especially,
     - Choice of human trajectory predictor: 
       - Set `sim.predict_method = 'inferred'` if a learning-based GST predictor is used [2]. Please also change `pred.model_dir` to be the directory of a trained GST model. We provide two pretrained models [here](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph/tree/main/gst_updated/results/).
       - Set `sim.predict_method = 'const_vel'` if constant velocity model is used.
       - Set `sim.predict_method = 'truth'` if ground truth predictor is used.
       - Set `sim.predict_method = 'none'` if you do not want to use future trajectories to change the observation and reward.
     - Randomization of human behaviors: If you want to randomize the ORCA humans, 
       - set `env.randomize_attributes` to True to randomize the preferred velocity and radius of humans;
       - set `humans.random_goal_changing` to True to let humans randomly change goals before they arrive at their original goals.

  2. PPO and network configurations: modify `arguments.py`
     - `env_name` (must be consistent with `sim.predict_method` in `crowd_nav/configs/config.py`): 
        - If you use the GST predictor, set to `CrowdSimPredRealGST-v0`.
        - If you use the ground truth predictor or constant velocity predictor, set to `CrowdSimPred-v0`.
        - If you don't want to use prediction, set to `CrowdSimVarNum-v0`. 
     - `use_self_attn`: human-human attention network will be included if set to True, else there will be no human-human attention.
     - `use_hr_attn`: robot-human attention network will be included if set to True, else there will be no robot-human attention.
- After you change the configurations, run
  ```
  python train.py 
  ```
- The checkpoints and configuration files will be saved to the folder specified by `output_dir` in `arguments.py`.

### Testing
Please modify the test arguments in line 20-33 of `test.py` (**Don't set the argument values in terminal!**), and run   
```
python test.py 
```
Note that the `config.py` and `arguments.py` in the testing folder will be loaded, instead of those in the root directory.  
The testing results are logged in `trained_models/your_output_dir/test/` folder, and are also printed on terminal.  
If you set `visualize=True` in `test.py`, you will be able to see visualizations like this:   

<div align="visual">
<img src=./figures/figure3.gif width=40% />
</div>


## Disclaimer
We only tested our code in Ubuntu with Python 3.11.4. The code may work on other OS or other versions of Python, but we do not have any guarantee.


## Citation
If you find the code or the paper useful for your research, please cite the following papers:
```
@Article{electronics12234744,
AUTHOR = {Sun, Xueying and Zhang, Qiang and Wei, Yifei and Liu, Mingmin},
TITLE = {Risk-Aware Deep Reinforcement Learning for Robot Crowd Navigation},
JOURNAL = {Electronics},
VOLUME = {12},
YEAR = {2023},
NUMBER = {23},
ARTICLE-NUMBER = {4744},
URL = {https://www.mdpi.com/2079-9292/12/23/4744},
ISSN = {2079-9292},
ABSTRACT = {Ensuring safe and efficient navigation in crowded environments is a critical goal for assistive robots. Recent studies have emphasized the potential of deep reinforcement learning techniques to enhance robots&rsquo; navigation capabilities in the presence of crowds. However, current deep reinforcement learning methods often face the challenge of robots freezing as crowd density increases. To address this issue, a novel risk-aware deep reinforcement learning approach is proposed in this paper. The proposed method integrates a risk function to assess the probability of collision between the robot and pedestrians, enabling the robot to proactively prioritize pedestrians with a higher risk of collision. Furthermore, the model dynamically adjusts the fusion strategy of learning-based and risk-aware-based features, thereby improving the robustness of robot navigation. Evaluations were conducted to determine the effectiveness of the proposed method in both low- and high-crowd density settings. The results exhibited remarkable navigation success rates of 98.0% and 93.2% in environments with 10 and 20 pedestrians, respectively. These findings emphasize the robust performance of the proposed method in successfully navigating through crowded spaces. Additionally, the approach achieves navigation times comparable to those of state-of-the-art methods, confirming its efficiency in accomplishing navigation tasks. The generalization capability of the method was also rigorously assessed by subjecting it to testing in crowd environments exceeding the training density. Notably, the proposed method attains an impressive navigation success rate of 90.0% in 25-person environments, surpassing the performance of existing approaches and establishing itself as a state-of-the-art solution. This result highlights the versatility and effectiveness of the proposed method in adapting to various crowd densities and further reinforces its applicability in real-world scenarios.},
DOI = {10.3390/electronics12234744}
}

```


Part of the code is based on the following repositories:  

[1] Liu S, Chang P, Huang Z, et al. Intention aware robot crowd navigation with attention-based interaction graph[C]//2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023: 12015-12021. (Github: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph?tab=readme-ov-file)

[2] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell, "Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning," in IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 3517-3524. (Github: https://github.com/Shuijing725/CrowdNav_DSRNN)  

[3] Z. Huang, R. Li, K. Shin, and K. Driggs-Campbell. "Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction," in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1198–1205, 2022. (Github: https://github.com/tedhuang96/gst)  

