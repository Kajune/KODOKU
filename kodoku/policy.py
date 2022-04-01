from typing import *
import numpy as np
from abc import ABCMeta, abstractmethod

import ray
from ray.rllib.evaluation.episode import Episode


class PolicyMappingManager(metaclass=ABCMeta):
	@abstractmethod
	def get_policy_mapping(self, policy_id : str, episode :Episode) -> str:
		""" Subpolicy to policy mapping function
		
		Args:
		    policy_id (str): Policy name
		    episode (Episode): Episode instance
		"""
		raise NotImplementedError()


	@abstractmethod
	def get_policy_mapping_list(self, policy_id : str) -> List[str]:
		""" Get all possible policy mapping
		
		Args:
		    policy_id (str): Policy name
		"""
		raise NotImplementedError()


	@abstractmethod
	def update_policy_mapping(self, reward_list : List[Dict]) -> None:
		""" Update policy mapping according to training result, if necessary
		
		Args:
		    reward_list (List[Dict]): Reward list by policy
		"""
		raise NotImplementedError()



class DefaultPolicyMappingManager(PolicyMappingManager):
	def get_policy_mapping(self, policy_id : str) -> str:
		return policy_id


	def get_policy_mapping_list(self, policy_id : str, episode :Episode) -> List[str]:
		return [policy_id]


	def update_policy_mapping(self, reward_list : List[Dict]) -> None:
		return


class SelfPlayPolicyMappingManager(PolicyMappingManager):
	def __init__(self, num_subpolicies : int, uniform_in_episode : bool = True):
		self.num_subpolicies = num_subpolicies
		self.uniform_in_episode = uniform_in_episode
		self.episode_policy_mapping = {}


	def get_policy_mapping(self, policy_id : str, episode : Episode) -> str:
		if self.uniform_in_episode:
			if (episode.episode_id, policy_id) not in self.episode_policy_mapping:
				self.episode_policy_mapping[(episode.episode_id, policy_id)] = \
					policy_id + '_' + str(np.random.randint(self.num_subpolicies))

			return self.episode_policy_mapping[(episode.episode_id, policy_id)]

		else:
			return policy_id + '_' + str(np.random.randint(self.num_subpolicies))


	def get_policy_mapping_list(self, policy_id : str) -> List[str]:
		return [policy_id + '_' + str(i) for i in range(self.num_subpolicies)]


	def update_policy_mapping(self, reward_list : List[Dict]) -> None:
		return

