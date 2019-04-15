# [Markov Decision Process](https://github.com/carol-hsu/mdp_study)

## Environment setup
Make sure you have Python3 on your machine.
After you pull this repo, following commands helps you installing the required packages.

```
// in Ubuntu 18.04, install following packages
$ sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev 

// virtual environment is recommended, but optional
$ virtualenv venv -p python3

(venv) $ pip install -r requirements.txt

```
## Run MDP cases with basic planners or reinforcement algorithms

There are three input parameters you can make adjustment: which case to run, run with which methods and set the discount value.

```
// cheking help message for details
(venv) $ python cloud_learner.py --help
usage: cloud_learner.py [-h] [-c CASE] [-d DISCOUNT] [-o OPERATION]

optional arguments:
  -h, --help            show this help message and exit
  -c CASE, --case CASE  Which case to run [0] machine-only-state case [1]
                        machine-with-customer-state case (default: 0)
  -d DISCOUNT, --discount DISCOUNT
                        Discount for value iteration and policy iteration
                        (default: 0.96)
  -o OPERATION, --operation OPERATION
                        Applying which method to find policy [0] value
                        iteration [1] policy iteration [2] RL algo (default:
                        0)
```



