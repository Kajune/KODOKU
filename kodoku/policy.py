from typing import *
from abc import ABCMeta, abstractmethod
import numpy as np
import random
import re

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
	def update_policy_configuration(self, trainer : "KODOKUTrainer", reward_list : List[Dict]) -> None:
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


	def update_policy_configuration(self, trainer : "KODOKUTrainer", reward_list : List[Dict]) -> None:
		return


class MultiSelfPlayPolicyMappingManager(PolicyMappingManager):
	def __init__(self, 
		blufor_policies : List[str],
		redfor_policies : List[str],
		num_blufor_subpolicies : int,
		num_redfor_subpolicies : int,
	):
		self.blufor_policies = blufor_policies
		self.redfor_policies = redfor_policies
		self.num_blufor_subpolicies = num_blufor_subpolicies
		self.num_redfor_subpolicies = num_blufor_subpolicies
		self.episode_policy_mapping : Dict[int,Tuple[int,int]] = {}
		self.match_results = [[[] for ri in range(self.num_redfor_subpolicies)] for bi in range(self.num_blufor_subpolicies)]


	def policy_selection(self):
		return {
			"blufor": np.random.randint(self.num_blufor_subpolicies), 
			"redfor": np.random.randint(self.num_redfor_subpolicies)
		}


	def policy_name(self, policy_id : str, index : int, force : str) -> str:
		return policy_id + '_' + str(index) + '_' + force


	def get_policy_mapping(self, policy_id : str, episode : Episode) -> str:
		if episode.episode_id not in self.episode_policy_mapping:
			self.episode_policy_mapping[episode.episode_id] = self.policy_selection()

		if policy_id in self.blufor_policies:
			return self.policy_name(policy_id, self.episode_policy_mapping[episode.episode_id]["blufor"], "blufor")
		elif policy_id in self.redfor_policies:
			return self.policy_name(policy_id, self.episode_policy_mapping[episode.episode_id]["redfor"], "redfor")
		else:
			print("Policy %s is not registered in SelfPlayPolicyMappingManager." % policy_id)


	def get_policy_mapping_list(self, policy_id : str) -> List[str]:
		if policy_id in self.blufor_policies:
			return [self.policy_name(policy_id, i, "blufor") for i in range(self.num_blufor_subpolicies)]
		elif policy_id in self.redfor_policies:
			return [self.policy_name(policy_id, i, "redfor") for i in range(self.num_redfor_subpolicies)]
		else:
			print("Policy %s is not registered in SelfPlayPolicyMappingManager." % policy_id)


	def update_policy_configuration(self, trainer : "KODOKUTrainer", reward_list : List[Dict]) -> None:
		current_match_results = self.compute_match_results(reward_list, 
			ma_length = len(reward_list) // (self.num_blufor_subpolicies * self.num_redfor_subpolicies))


	def compute_match_results(self, reward_list : List[Dict], ma_length : int = 1000) -> np.ndarray:
		# Collect performance results of each subpolicy from trajectory
		for episode_id, reward_dict in reward_list.items():
			blufor_subpolicy_index = self.episode_policy_mapping[episode_id]["blufor"]
			redfor_subpolicy_index = self.episode_policy_mapping[episode_id]["redfor"]

			blufor_reward_list = []
			redfor_reward_list = []
			for (_, subpolicy_id), reward in reward_dict.items():
				for policy_id in self.blufor_policies:
					if self.policy_name(policy_id, blufor_subpolicy_index, "blufor") == subpolicy_id:
						blufor_reward_list.append(reward)
						break

				for policy_id in self.redfor_policies:
					if self.policy_name(policy_id, redfor_subpolicy_index, "redfor") == subpolicy_id:
						redfor_reward_list.append(reward)
						break

			self.match_results[blufor_subpolicy_index][redfor_subpolicy_index].append(np.float32([np.mean(blufor_reward_list), np.mean(redfor_reward_list)]))

		# Compute moving average
		current_match_results = np.zeros((self.num_blufor_subpolicies, self.num_redfor_subpolicies, 2), dtype=np.float32)

		for bi in range(self.num_blufor_subpolicies):
			for ri in range(self.num_redfor_subpolicies):
				if len(self.match_results[bi][ri]) == 0:
					print('Match result computation failed due to insufficient samples.')
					return None
		
				if len(self.match_results[bi][ri]) <= ma_length:
					current_match_results[bi][ri] = np.mean(self.match_results[bi][ri], axis=0)
				else:
					current_match_results[bi][ri] = np.mean(self.match_results[bi][ri][-ma_length:], axis=0)

		return current_match_results
