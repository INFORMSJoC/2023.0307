[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# An Efficient Node Selection Policy for Monte Carlo Tree Search with Neural Networks
This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE.txt).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper
[An Efficient Node Selection Policy for Monte Carlo Tree Search with Neural Networks](https://doi.org/10.1287/ijoc.2023.0307) by Xiaotian Liu, Yijie Peng, Gongbo Zhang, and Ruihan Zhou.


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0307

https://doi.org/10.1287/ijoc.2023.0307.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{liu:2024,
  author =        {Liu, Xiaotian and Peng, Yijie and Zhang, Gongbo and Zhou, Ruihan},
  publisher =     {INFORMS Journal on Computing},
  title =         {{An Efficient Node Selection Policy for Monte Carlo Tree Search with Neural Networks}},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0307.cd},
  url =           {https://github.com/INFORMSJoC/2023.0307},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0307},
}  
```

## Description

This directory contains the folders `src` and `data`:
- `src`: includes the source codes of the paper.
  - `src/AOAT-MCTS-Tic-Tac-Toe`: codes for UCT/OCBA-MCTS/AOAT-MCTS/ implemented on Tic-Tac-Toe
  - `src/AOAT-NN-Board-Games`: codes for AOAT implemented with NNs applied on board games
  - `src/AOAT-NN-Classical-Control`: codes for UCT/AOAT implemented with NNs applied on Cartpole
  - ===========Your part of codes here================
  
- `results`: contains results presented in paper.
  - `src/results`: results related to AOAT implemented with NNs applied on board games
  - ===========Your part of codes here================

## Dependencies

- For codes under `src/AOAT-MCTS-Tic-Tac-Toe`:
  - `python 3.8`
- For codes under `src/AOAT-NN-Board-Games` and `src/AOAT-NN-General-RL-Tasks`:
  - `python 3.8`
  - `pytorch 1.8.1`
-  ===========Your part of codes here================

## Run experiments

**1. Run experiments related to AOAP-MCTS compard with OCBA-MCTS on Tic-Tac-Toe**

First cd into folder `src/AOAT-MCTS-Tic-Tac-Toe`

Run `tic_tac_toe.py` to obtain resutls

**2. Run experiments related to AOAT implemented with NNs applied on board games**

First cd into folder `src/AOAT-NN-Board-Games`

- Train NNs

The main logic of the training process is shown in the following figure

<img src="https://github.com/xiaotianliu01/AOAP-Value-Network-MCTS/blob/master/diagram.png" width="400" height="300">

For each iteration, the python file *Simulate.py* is used to simulate the games to collect training data, and the python file *Learn.py* is used to train the NN models with the collected training data for one iteration.

***use the script *train.sh* to automatically do the training for multiple iterations***
```Bash
sh train.sh
```
where you can specify the number of iterations by modifying the file *train.sh*.

All exgeneous parameters are determined in the file *config.py*. The default parameters are used for generating results of Tic-tac-toe game

After running the training script, a new folder `\temp` will be created, which stores obtained models for each iteration. Under this folder, '\Iter1' contains models and data for iteration 1, '\Iter2' contains models and data for iteration 2, ...

***To compete different forms of AOAT with UCT by using the obtained NNs, run***
```Bash
python3 pit.py 1
```
where the argument 1 is the random seed.

Modify parameters in the python file `pit.py` to specify AOAT form and considered parameters. After runing this python file, mutiple txt files will be created which contain competing results including number of winning games for each policy. 

**3. Run experiments related to AOAT implemented with NNs applied on general RL tasks**

### Usage:
* Train: ```python main.py --env CartPole-v0/v1 --opr train --force ```
* Test: ```python main.py --env CartPole-v0/v1 ---opr test```
  
| Required Arguments   | Description                                           |
|:---------------------|:------------------------------------------------------|
| `--env`              | Name of the environment  (CartPole-v0 or CartPole-v1) |
| `--opr {train,test}` | select the operation to be performed                  |

* Node selection policy modification： ```src/AOAT-NN-General-RL-Tasks/core/config.py```
  
*CartPole parameter modification: ```src/AOAT-NN-General-RL-Tasks/config/cartpole/_init_.py```

