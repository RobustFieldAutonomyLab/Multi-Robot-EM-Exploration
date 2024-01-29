# Distributional RL for Navigation
This is the implementation of our autonomous exploration algorithm designed for decentralized multi-robot teams, which takes into account map and localization uncertainties of range-sensing
mobile robots. Virtual landmarks are used to quantify the combined impact of process noise and sensor noise on map uncertainty. Additionally, we employ an iterative expectation-maximization inspired algorithm to assess the potential outcomes of both a local robot’s and its neighbors’ next-step actions.The Environment used in this repo is from our IROS 2023 paper [here](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Navigation).

<p align="center"><img src="real_pipeline.jpeg" width=700></p>

