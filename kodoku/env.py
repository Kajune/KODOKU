from typing import *
from abc import ABCMeta, abstractmethod
from gym import Env, spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode


class EnvWrapper(MultiAgentEnv, metaclass=ABCMeta):
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
	def get_spaces(self) -> Dict[str, Tuple[spaces.Space, spaces.Space]]:
		""" Get dictionary with agent name key and  observation and action spaces value
		"""
		raise NotImplementedError()


	@abstractmethod
	def get_policy_mapping_fn(self) -> Callable[[str, Episode], str]:
		"""Get policy mapping for multiagetn training
		"""
		raise NotImplementedError()



	@abstractmethod
	def get_obs(self) -> Dict[str, Any]:
		""" reset
		
		Returns:
		    Dict[str, Any]: dictionary with agent name key and observation value
		"""
		raise NotImplementedError()


	@abstractmethod
	def step(self, action : Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict]:
		""" step
		
		Args:
		    action (Dict[str, Any]): dictionary with agent name key and action value
		"""
		raise NotImplementedError()


	def render(self, mode : str = 'human') -> Any:
		""" render
		
		Args:
		    mode (str, optional): 'rgb_array' or 'human' or 'ansi'
		"""
		raise NotImplementedError()


	def reset(self) -> Dict[str, Any]:
		""" reset
		
		Returns:
		    Dict[str, Any]: dictionary with agent name key and observation value
		"""
		self.scenario_name, self.config = self.config_fn()
		self.env = self.initialize_env(self.config)
		return self.get_obs()
