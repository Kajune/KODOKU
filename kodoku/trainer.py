from typing import *
from gym import Env, spaces
import numpy as np

import ray
import ray.rllib.agents as agents
from ray.rllib.agents.trainer import Trainer
from ray.tune.logger import pretty_print
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.utils.typing import ResultDict

from torch.utils.tensorboard import SummaryWriter

from kodoku.env import EnvWrapper
from kodoku.policy import *
from kodoku.utils import LogCallbacks, print_network_architecture


class KODOKUTrainer:
	def __init__(self, 
		log_dir : str, 
		env_class : Type[EnvWrapper],
		train_config : Dict,
		env_config_fn : Callable[[], Tuple[str, Dict]],
		policy_mapping_manager : Optional[PolicyMappingManager] = None,
		ray_config : Dict = {}):
		""" Trainer ctor
		
		Args:
		    log_dir (str): Location to store training summary, trajectory, weight, etc..
		    env_class (Type[EnvWrapper]): Environment class
		    train_config (Dict): Training config
		    env_config_fn (Callable[[], Tuple[str, Dict]]): Functor to set config on env
		    policy_mapping_manager (Optional[PolicyMappingManager], optional): Policy mapping manager instance
		    ray_config (Dict, optional): ray configuration for init (memory, num gpu, etc..)
		"""

		# Ray initialization
		ray.init(**ray_config)
		assert ray.is_initialized() == True

		self.summaryWriter = SummaryWriter(log_dir)

		# Training configuration
		trainer_class, self.train_config = agents.registry.ALGORITHMS[train_config['algorithm']]()
		for group in ["general", "environment", "training"]:
			for k, v in train_config[group].items():
				self.train_config[k] = v

		# Set lr scheduler for wolf
		if self.train_config["lr_schedule"] is None:
			self.train_config["lr_schedule"] = [[0,self.train_config["lr"]],[1,self.train_config["lr"]]]

		if self.train_config["model"] == "default":
			self.train_config["model"] = MODEL_DEFAULTS

		self.train_config["callbacks"] = LogCallbacks

		# Environment configuration
		self.train_config["env"] = env_class
		self.train_config["env_config"] = { "fn": env_config_fn }

		# Multiagent configuration
		for k, v in train_config["multiagent"].items():
			self.train_config["multiagent"][k] = v

		tmp_env = self.train_config["env"](self.train_config["env_config"])
		policy_mapping_fn_tmp = tmp_env.get_policy_mapping_fn()

		self.policy_mapping_manager = policy_mapping_manager

		if self.policy_mapping_manager is None:
			self.train_config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn_tmp
			self.train_config["multiagent"]["policies"] = { 
				policy_name: (None, obs_space, act_space, {}) 
					for policy_name, (obs_space, act_space) in tmp_env.get_spaces().items() 
			}

		else:
			def policy_mapping_fn(agent_id : str, episode : Episode, **kwargs) -> str:
				return self.policy_mapping_manager.get_policy_mapping(
					agent_id, policy_mapping_fn_tmp(agent_id, episode, **kwargs), episode)

			self.train_config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn
			self.train_config["multiagent"]["policies"] = { 
				subpolicy_name: (None, obs_space, act_space, {}) 
					for policy_name, (obs_space, act_space) in tmp_env.get_spaces().items() 
						for subpolicy_name in self.policy_mapping_manager.get_policy_mapping_list(policy_name) 
			}

		# Initialize trainer
		self.trainer = trainer_class(config=self.train_config)

		print_network_architecture(self.trainer, self.train_config["multiagent"]["policies"].keys())


	def train(self, 
		num_epochs : int, 
		start_epoch : int = 0,
		epoch_callback : Optional[Callable[["KODOKUTrainer", int, ResultDict], None]] = None) -> None:
		""" Run train
		
		Args:
		    num_epochs (int): Number of epochs to train
		    start_epoch (int, optional): Start epoch, if you want to start from intermediate
		    epoch_callback (Optional[Callable[["KODOKUTrainer", int, ResultDict], None]], optional): Callback function called on each epoch
		"""

		for epoch in range(start_epoch, num_epochs):
			self.trainer.callbacks.reset()
			result = self.trainer.train()

			# Update policy mapping
			if self.policy_mapping_manager is not None:
				self.policy_mapping_manager.update_policy_configuration(self.trainer, self.reward())

			# Invoke callback
			if epoch_callback is not None:
				epoch_callback(self, epoch, result)

			print(pretty_print(result))

			# write log to tensorboard
			for policy in result["policy_reward_mean"]:
				if policy in result["info"]["learner"]:
					for k, v in result["info"]["learner"][policy]["learner_stats"].items():
						self.summaryWriter.add_scalar(k + '_' + policy, v, epoch)
				for k in ['mean', 'min', 'max']:
					self.summaryWriter.add_scalar('EpRet_' + policy + '_' + k, result['policy_reward_' + k][policy], epoch)


	def evaluate(self) -> ResultDict:
		""" Run evaluation
		
		Returns:
		    ResultDict: Evaluation result
		"""
		self.trainer.callbacks.reset()
		result = self.trainer.evaluate()
		print(pretty_print(result))
		return result


	def log(self) -> Dict:
		""" Get training log
		
		Returns:
		    Dict: log
		"""
		return self.trainer.callbacks.log()


	def reward(self) -> Dict[int,Dict]:
		""" Get training rewards

		Returns:
			List: reward
		"""
		return self.trainer.callbacks.reward()
		
