# Where to go next: Learning a Subgoal Recommendation Policy for Navigation Among Pedestrians

<img src="figs/000_MPCRLPolicy_8agents.gif" alt="">

This repository will contain the code for the paper:

**<a href="https://arxiv.org/pdf/2102.13073.pdf">Where to go next: Learning a Subgoal Recommendation Policy for Navigation Among Pedestrians</a>**
<br>
<a href="http://www.tudelft.nl/staff/bruno.debrito/">Bruno Brito</a>,
<a href="http://mfe.mit.edu/">Michael Everett</a>,
<a href="http://www.mit.edu/~jhow/">Jonathan P. How</a>,
<a href="http://www.tudelft.nl/staff/j.alonsomora/">Javier Alonso-Mora</a>
<br>
Accepted for publication in [RA-L + ICRA 2021].

Robotic navigation in environments shared with other robots or humans remains challenging because the intentions of the surrounding agents are not directly observable and the environment conditions are continuously changing. 
Local trajectory optimization methods, such as model predictive control (MPC), can deal with those changes but require global guidance, which is not trivial to obtain in crowded scenarios.
This paper proposes to learn, via deep Reinforcement Learning (RL), an interaction-aware policy that provides long-term guidance to the local planner. In particular, in simulations with cooperative and non-cooperative agents, we train a deep network to recommend a subgoal for the MPC planner. The recommended subgoal is expected to help the robot in making progress towards its goal and accounts for the expected interaction with other agents. Based on the recommended subgoal, the MPC planner then optimizes the inputs for the robot satisfying its kinodynamic and collision avoidance constraints.
Our approach is shown to substantially improve the navigation performance in terms of
number of collisions as compared to prior MPC frameworks, and in terms of both travel time and number of collisions compared to deep RL methods in cooperative, competitive and mixed multiagent scenarios.

<div align='center'>
<img src="figs/072_MPCRLPolicy_10agents.gif"></img>
</div>

If you find this code useful in your research then please cite
```
@article{britogompc,
  title={Where to go next: Learning a Subgoal Recommendation Policy for Navigation Among Pedestrians},
  author={Brito, Bruno and Everett, Michael and How, Jonathan and Alonso-Mora, Javier},
  journal={IEEE Robotics and Automation Letters},
  year={2021},
  publisher={IEEE}
}
```

This is a pre-release of the code used for the publication above. Please note that this code was not cleanup or fully tested yet. We plan to make a cleanup release soon. Moreover, for the training algorithm we heavily rely on **<a href="https://stable-baselines.readthedocs.io/en/master/">StableBaselines 2.0</a>**.

Installation
========================

Grab the simulation environment from github, initialize submodules, install dependencies and src code:

```
    git clone --recursive --branch ral2021 https://github.com/bbrito/gym-collision-avoidance.git
    cd gym-collision-avoidance
    ./install.sh
```

Download the GO-MPC source code:

```
    git clone https://github.com/tud-amr/go-mpc.git
```

Generate the MPC solver using Matlab. You will need a ForcesPro license to generate the solver. You can request it here **<a href="https://www.embotech.com/products/forcespro/licensing/">FORCESPRO</a>**. 
Once you get the solver license:
```
    Go to the directory  <go-mpc directoy>/mpc_rl_collision_avoidance/mpc
    Run scenario.m
```

Start training:
```
    source <gym-collision-avoidance directory>/venv/bin/activate
    cd <go-mpc directoy>
    python setup.py install
    python train.py
```

To test the trained network do:
```
    cd <go-mpc directoy>
    ./test.sh
```

To download a pre-trained model do:
```
    cd <go-mpc directoy>
    mkdir logs/ppo2-mpc 
    ./download_trained_models.sh
```

All the training parameters such as reward function, scenarios used for training and network policy are defined in the following two files:
```
    <go-mpc directoy>/mpc_rl_collision_avoidance/hyperparams/ppo2-mpc.yml
    <gym-collision-avoidance directory>/gym_collision_avoidance/envs/config.py
```

The main GO-MPC traning algorithm is defined in:
```
    <go-mpc directoy>/mpc_rl_collision_avoidance/algorithms/ppo2/ppo2mpc.py
```

The network policy is defined with the name MlpLstmPolicy and is defined in:
```
    <go-mpc directoy>/mpc_rl_collision_avoidance/external/stable_baselines/common/policies.py
```
The LSTM dynamic layer in:
```
    <go-mpc directoy>/mpc_rl_collision_avoidance/external/stable_baselines/common/tf_layers.py
```
