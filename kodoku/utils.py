from typing import *
from gym import Env, spaces
import numpy as np
from numba import njit

import ray
from ray.tune.logger import pretty_print
from ray.rllib.env import BaseEnv
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.policy import Policy
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.typing import AgentID, PolicyID, TensorType
from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.schedules.schedule import Schedule

from kodoku.env import EnvWrapper



class LogCallbacks(DefaultCallbacks):
	log_dict = {}
	reward_dict = {}

	def __init__(self):
		super().__init__()
		self.reset()


	def log(self) -> Dict:
		return LogCallbacks.log_dict


	def reward(self) -> Dict[int,Dict]:
		return LogCallbacks.reward_dict


	def reset(self) -> None:
		LogCallbacks.log_dict = {}
		LogCallbacks.reward_dict = {}


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
		LogCallbacks.reward_dict[episode.episode_id] = episode.agent_rewards


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
			if isinstance(policy, TorchPolicy):
				print(policy.model)
			elif isinstance(policy, TFPolicy):
				policy.model.base_model.summary()
			else:
				print('Unknown framework:', policy)
		else:
			print('Policy for %s is None' % policy_name)



class ScheduleScaler(Schedule):
	def __init__(self, schedule : Schedule, scale : float = 1.0):
		""" Schedule scaler
		This class wraps existing schedule instance to scale its value
		
		Args:
		    schedule (Schedule): Schedule instance
		    scale (float, optional): Scale
		"""
		self.schedule = schedule
		self.scale = scale
		self.framework = schedule.framework

	
	def _value(self, t: Union[int, TensorType]) -> Any:
		return self.schedule(t) * self.scale
		