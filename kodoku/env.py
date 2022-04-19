from typing import *
from abc import ABCMeta, abstractmethod

import gym
from gym import Env, spaces

from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode



class EnvWrapper(metaclass=ABCMeta):
	reward_range = (-float('inf'), float('inf'))
	observation_space = None
	action_space = None

	def __init__(self, config : EnvContext):
		""" EnvWrapper ctor
		
		Args:
		    env_fn (Callable[[Dict], Any]): Functor to generate env object
		    config_fn (Callable[[], [str, Dict]]): Functor to generate config for env
		"""
		super().__init__()

		self.config_fn = config["fn"]

		self.env = None
		self.scenario_name = None
		self.config = None


	@abstractmethod
	def log(self) -> Dict:
		""" Generate dictionary to log trajectory
		"""
		raise NotImplementedError()


	@abstractmethod
	def initialize_env(self, config : Dict) -> Any:
		""" Env factory function
		
		Args:
		    config (Dict): Env config
		"""
		raise NotImplementedError()


	@abstractmethod
	def get_spaces(self) -> Union[Tuple[spaces.Space, spaces.Space], Dict[str, Tuple[spaces.Space, spaces.Space]]]:
		""" Get dictionary with agent name key and  observation and action spaces value
		"""
		raise NotImplementedError()


	@abstractmethod
	def step(self, action : Any) -> Union[Tuple[Any, float, bool, Dict], Dict[str, Tuple[Any, float, bool, Dict]]]:
		""" step
		
		Args:
		    action (Dict[str, Any]): dictionary with agent name key and action value
		"""
		raise NotImplementedError()


	@abstractmethod
	def reset_impl(self) -> Union[Any, Dict[str, Any]]:
		""" reset_impl
		"""
		raise NotImplementedError()


	def render(self, mode : str = 'human') -> Any:
		""" render
		
		Args:
		    mode (str, optional): 'rgb_array' or 'human' or 'ansi'
		"""
		raise NotImplementedError()


	def reset(self) -> Union[Any, Dict[str, Any]]:
		""" reset
		
		Returns:
		    Union[Any, Dict[str, Any]]: dictionary with agent name key and observation value
		"""
		self.scenario_name, self.config = self.config_fn()
		self.env = self.initialize_env(self.config)
		return self.reset_impl()



class MultiEnvWrapper(EnvWrapper, MultiAgentEnv, metaclass=ABCMeta):
	@abstractmethod
	def get_policy_mapping_fn(self) -> Callable[[str, Episode], str]:
		"""Get policy mapping for multiagetn training
		"""
		raise NotImplementedError()



class GymEnv(EnvWrapper):
	def log(self) -> Dict:
		return {}


	def initialize_env(self, config : Dict) -> Any:
		env = gym.make(**config)
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.reward_range = env.reward_range
		return env


	def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
		if self.env is None:
			self.env = self.initialize_env(self.config_fn()[1])			
		return self.env.observation_space, self.env.action_space


	def step(self, action : Any) -> Tuple[Any, float, bool, Dict]:
		return self.env.step(action)


	def reset_impl(self) -> Any:
		return self.env.reset()


	def render(self, mode : str = 'rgb_array') -> Any:
		return self.env.render(mode)
