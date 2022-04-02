from typing import *
from abc import ABCMeta, abstractmethod
import numpy as np
import random
import re

import ray
from ray.rllib.evaluation.episode import Episode



class PolicyMappingManager(metaclass=ABCMeta):
	@abstractmethod
	def get_policy_mapping(self, agent_id : str, policy_id : str, episode :Episode) -> str:
		""" Subpolicy to policy mapping function
		
		Args:
		    agent_id (str): Agent name
		    policy_id (str): Policy name
		    episode (Episode): Episode instance
		
		Raises:
		    NotImplementedError: Description
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
	def update_policy_configuration(self, trainer : "KODOKUTrainer", reward_list : List[Dict]) -> None:
		""" Update policy mapping according to training result, if necessary
		
		Args:
		    reward_list (List[Dict]): Reward list by policy
		"""
		raise NotImplementedError()



class DefaultPolicyMappingManager(PolicyMappingManager):
	def get_policy_mapping(self, agent_id : str, policy_id : str) -> str:
		return policy_id


	def get_policy_mapping_list(self, policy_id : str, episode :Episode) -> List[str]:
		return [policy_id]


	def update_policy_configuration(self, trainer : "KODOKUTrainer", reward_list : List[Dict]) -> None:
		return



class SelfPlayManager(PolicyMappingManager):
	def __init__(self, 
		agent_force_mapping_fn : Callable[[str], str],
		num_subpolicies : int,
	):
		""" SelfPlayManager
		
		Args:
		    agent_force_mapping_fn (Callable[[str], str]): Function to determine which force an agent belongs to. Must return "blufor" or "redfor".
		    num_subpolicies (int): Number of subpolicies
		"""
		self.agent_force_mapping_fn = agent_force_mapping_fn
		self.num_subpolicies = num_subpolicies
		self.episode_policy_mapping : Dict[int,Tuple[int,int]] = {}
		self.episode_agent_mapping : Dict[int,Dict[str,str]] = {}
		self.match_results = [[[] for ri in range(self.num_subpolicies)] for bi in range(self.num_subpolicies)]


	def policy_selection(self):
		return {
			"blufor": np.random.randint(self.num_subpolicies), 
			"redfor": np.random.randint(self.num_subpolicies)
		}


	def policy_name(self, policy_id : str, index : int) -> str:
		return policy_id + '_' + str(index)


	def get_policy_mapping(self, agent_id : str, policy_id : str, episode : Episode) -> str:
		if episode.episode_id not in self.episode_policy_mapping:
			self.episode_policy_mapping[episode.episode_id] = self.policy_selection()
			self.episode_agent_mapping[episode.episode_id] = {}

		force = self.agent_force_mapping_fn(agent_id)
		assert force == "blufor" or force == "redfor", "agent_force_mapping_fn must return \"blufor\" or \"redfor\"."
		self.episode_agent_mapping[episode.episode_id][agent_id] = force
		return self.policy_name(policy_id, self.episode_policy_mapping[episode.episode_id][force])


	def get_policy_mapping_list(self, policy_id : str) -> List[str]:
		return [self.policy_name(policy_id, i) for i in range(self.num_subpolicies)]


	def update_policy_configuration(self, trainer : "KODOKUTrainer", reward_list : List[Dict]) -> None:
		current_match_results = self.compute_match_results(reward_list, 
			ma_length = len(reward_list) // (self.num_subpolicies ** 2))
		if current_match_results is not None:
			print((current_match_results[...,0] - current_match_results[...,1]) / 2)


	def compute_match_results(self, reward_list : List[Dict], ma_length : int = 1000) -> np.ndarray:
		# Collect performance results of each subpolicy from trajectory
		for episode_id, reward_dict in reward_list.items():
			blufor_subpolicy_index = self.episode_policy_mapping[episode_id]["blufor"]
			redfor_subpolicy_index = self.episode_policy_mapping[episode_id]["redfor"]

			blufor_reward_list = []
			redfor_reward_list = []
			for (agent_id, subpolicy_id), reward in reward_dict.items():
				if self.episode_agent_mapping[episode_id][agent_id] == "blufor":
					blufor_reward_list.append(reward)
				elif self.episode_agent_mapping[episode_id][agent_id] == "redfor":
					redfor_reward_list.append(reward)
				else:
					print("Unknown force %s in SelfPlayManager" % self.episode_agent_mapping[episode_id][agent_id])

			self.match_results[blufor_subpolicy_index][redfor_subpolicy_index].append(np.float32([np.mean(blufor_reward_list), np.mean(redfor_reward_list)]))

		# Compute moving average
		current_match_results = np.zeros((self.num_subpolicies, self.num_subpolicies, 2), dtype=np.float32)

		for bi in range(self.num_subpolicies):
			for ri in range(self.num_subpolicies):
				if len(self.match_results[bi][ri]) == 0:
					print('Match result computation failed due to insufficient samples.')
					return None
		
				if len(self.match_results[bi][ri]) <= ma_length:
					current_match_results[bi][ri] = np.mean(self.match_results[bi][ri], axis=0)
				else:
					current_match_results[bi][ri] = np.mean(self.match_results[bi][ri][-ma_length:], axis=0)

		return current_match_results
