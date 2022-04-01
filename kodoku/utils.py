from typing import *
from gym import Env, spaces
import numpy as np

import ray
from ray.tune.logger import pretty_print
from ray.rllib.env import BaseEnv
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.evaluation.episode import Episode

from kodoku.env import EnvWrapper


class LogCallbacks(DefaultCallbacks):
	log_dict = {}
	reward_list = []

	def __init__(self):
		super().__init__()
		self.reset()


	def log(self) -> Dict:
		return LogCallbacks.log_dict


	def reward(self) -> List:
		return LogCallbacks.reward_list


	def reset(self) -> None:
		LogCallbacks.log_dict = {}
		LogCallbacks.reward_list = []


	def common_callback(self, base_env: BaseEnv, env_index: int = None, **kwargs):
		ei : int = env_index if env_index is not None else 0
		envs : List[EnvWrapper] = base_env.get_sub_environments()
		scenario_name : str = envs[ei].scenario_name

		if scenario_name not in LogCallbacks.log_dict:
			LogCallbacks.log_dict[scenario_name] = {}
		if ei not in LogCallbacks.log_dict[scenario_name]:
			LogCallbacks.log_dict[scenario_name][ei] = []

		return envs[ei], scenario_name, ei


	def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
						 policies: Dict[PolicyID, Policy], episode: Episode,
						 **kwargs) -> None:
		env, scenario_name, ei = self.common_callback(base_env, **kwargs)
		LogCallbacks.log_dict[scenario_name][ei].append([])
		LogCallbacks.log_dict[scenario_name][ei][-1].append(env.log())


	def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
					   policies: Dict[PolicyID, Policy], episode: Episode,
					   **kwargs) -> None:
		env, scenario_name, ei = self.common_callback(base_env, **kwargs)
		if len(LogCallbacks.log_dict[scenario_name][ei]) == 0:
			LogCallbacks.log_dict[scenario_name][ei].append([])
		LogCallbacks.log_dict[scenario_name][ei][-1].append(env.log())


	def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
					   policies: Dict[PolicyID, Policy], episode: Episode,
					   **kwargs) -> None:
		LogCallbacks.reward_list.append(episode.agent_rewards)


def print_network_architecture(trainer : Trainer, policies : List[str]) -> None:
	""" Print network architectures for policies
	
	Args:
	    trainer (Trainer): Trainer object
	    policies (List[str]): Policies to print
	"""
	for policy_name in policies:
		print(policy_name, "Network Architecture")
		policy = trainer.get_policy(policy_name)
		if policy is not None:
			print(policy.model)
		else:
			print('Policy for %s is None' % policy_name)

