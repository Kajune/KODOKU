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


def callback(trainer : SingleAgentTrainer, epoch : int, result : Dict):
	log = trainer.log()
	json.dump(log, open('./log_dir/latest_log.json', 'w'), indent=2)


if __name__ == '__main__':
	trainer = SingleAgentTrainer(
		log_dir='./log_dir', 
		env_class=GymEnv,
		env_config_fn=config_fn,
		train_config=json.load(open('train_config.json')),
	)

	trainer.train(10, epoch_callback=callback)
	trainer.evaluate()

