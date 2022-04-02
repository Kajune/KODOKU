from typing import *
import numpy as np
import json

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kodoku.trainer import KODOKUTrainer
from kodoku.policy import DefaultPolicyMappingManager, SelfPlayManager
from sampleEnv import SimpleBattlefieldEnv_Sym


def config_fn():
	return \
		'default', \
		{
			"depth": 2.0,
			"width": 1.0,
			"atk_spawn_line": 1.5,
			"def_spawn_line": 0.5,
			"atk_num" : 3,
			"def_num" : 3,
			"unit_hp" : 1.0,
			"unit_power": 0.1,
			"unit_range": 0.1,
			"unit_speed": 0.05,
			"timelimit": 500,
		}


def callback(trainer : KODOKUTrainer, epoch : int, result : Dict):
	log = trainer.log()
	json.dump(log, open('./log_dir/latest_log.json', 'w'), indent=2)


if __name__ == '__main__':
	trainer = KODOKUTrainer(
		log_dir='./log_dir', 
		env_class=SimpleBattlefieldEnv_Sym,
		train_config=json.load(open('train_config.json')),
		env_config_fn=config_fn,
		policy_mapping_manager=SelfPlayManager(lambda agent: "blufor" if agent.startswith("atk") else "redfor", 3),
	)

	trainer.train(10, epoch_callback=callback)
	trainer.evaluate()

