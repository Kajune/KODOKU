from typing import *
import numpy as np
import json

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kodoku.trainer import SingleAgentTrainer
from kodoku.env import GymEnv
import gym


def config_fn():
	return 'default', {'id': 'CartPole-v0'}


if __name__ == '__main__':
	trainer = SingleAgentTrainer(
		log_dir='./log_dir', 
		env_class=GymEnv,
		env_config_fn=config_fn,
		train_config=json.load(open('train_config.json')),
	)

	trainer.train(10)
	trainer.evaluate()

