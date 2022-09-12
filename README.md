# A Deep Reinforcement Learning Survey on Zero-Sum Games

<p align="center">
  <a href="https://boracchi.faculty.polimi.it/teaching/AdvancedDLMM.htm">
  <img src="https://miro.medium.com/max/1400/1*jbfAi9yVWv7J4FtdwSZnpw.png"/ alt="Advanced Deep Learning Models and Methods Course" style="width: 100%; height: 200px; object-fit: cover; object-position: 100% 0;">
  </a>
</p>


This repository contains the code written for the final project of the [Advanced Deep Learning Models and Methods Course](https://boracchi.faculty.polimi.it/teaching/AdvancedDLMM.htm) at [Politecnico di Milano](https://www.polimi.it/).
The repository is organized as follows
* [Agents](https://github.com/Menta99/Advanced_Deep_Learning_Models_and_Methods-Menta-Nisti/tree/master/Agents) which contains the Agents implemented in the project, namely [Dueling Double Deep Q-Network](https://arxiv.org/abs/1511.06581), [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) and [AlphaGo Zero](https://www.nature.com/articles/nature24270)
* [Utilities](https://github.com/Menta99/Advanced_Deep_Learning_Models_and_Methods-Menta-Nisti/tree/master/Utilities) which includes the infrastructure used for the tests, including the environment of the 3 implemented zero-sum games, specifically TicTacToe, ConnectFour and Santorini

Finally you can find an extended [report](https://github.com/Menta99/Advanced_Deep_Learning_Models_and_Methods-Menta-Nisti/blob/master/paper.pdf) of this work, containing the complete description of the approach and the results obtained, and a brief [presentation](https://github.com/Menta99/Advanced_Deep_Learning_Models_and_Methods-Menta-Nisti/blob/master/presentation.pptx) of it.

## Project Overview
The project consists in analyzig, reviewing and re-implementing 3 papers belonging to a field of application of Deep Learning of our choice, and in applying the techniques proposed in a novel context.

We have selected the Deep Reinforcement Learning field and the following 3 paper:

 - ["Dueling Network Architectures for Deep Reinforcement Learning"](https://arxiv.org/pdf/1511.06581.pdf)
 - ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/pdf/1801.01290.pdf)
 - ["Mastering the game of Go without human knowledge"](https://www.nature.com/articles/nature24270.pdf)
 
We have then applied these 3 approaches in the zero-sum games field. 

## Installation
The `requirements.txt` file list all Python libraries that this repository
depend on, and they will be installed using:

```
pip install -r requirements.txt
```
## Repository Structure
To reproduce the results you can use the [DataGenerator](https://github.com/Menta99/Advanced_Deep_Learning_Models_and_Methods-Menta-Nisti/blob/master/Utilities/DataGenerator.py) script:
You just need to include in the iterator the configurations to be tested, specifically you can choose*:
 - Agent (DDDQN, SAC, AlphaGo Zero)
 - Environment (TicTacToe, ConnectFour, Santorini)
 - Representation (Tabular, Graphic)
 - Opponent (Random, MinMaxRandom, MonteCarlo, Self)
 - Turn (First, Second, Random)

the script will take care of saving both the training data, the game GIFs and the model itself**.

To plot the resulting curves (game reward and game length) you can use the [Displayer](https://github.com/Menta99/Advanced_Deep_Learning_Models_and_Methods-Menta-Nisti/blob/master/Utilities/Displayer.py) script following the same configuration procedure as above.

<sup><sup>*Not all the combinations are supported due to the incompatible structure of agents and environments  
<sup><sup>**If you don't have a powerful GPU to train the models we suggest to use the [Notebook](https://github.com/Menta99/Advanced_Deep_Learning_Models_and_Methods-Menta-Nisti/blob/master/Utilities/TurnGameTesterNotebook.ipynb) we have prepared which contains both testing and visualization exploiting the [Google Colab](https://colab.research.google.com/) service

## Group Members
- [__Andrea Menta__](https://github.com/Menta99)
- [__Giovanni Nisti__](https://github.com/GiovanniN98)
