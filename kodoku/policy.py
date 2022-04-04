from typing import *
from abc import ABCMeta, abstractmethod
import numpy as np

import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.schedules.constant_schedule import ConstantSchedule

from kodoku.utils import ScheduleScaler



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
	def update_policy_configuration(self, trainer : Trainer, reward_list : List[Dict]) -> None:
		""" Update policy mapping according to training result, if necessary
		
		Args:
		    trainer (Trainer): Trainer instance
		    reward_list (List[Dict]): Reward list by policy		
		"""
		raise NotImplementedError()



class DefaultPolicyMappingManager(PolicyMappingManager):
	""" Identity policy mapping
	"""
	def get_policy_mapping(self, agent_id : str, policy_id : str, episode :Episode) -> str:
		return policy_id


	def get_policy_mapping_list(self, policy_id : str) -> List[str]:
		return [policy_id]


	def update_policy_configuration(self, trainer : Trainer, reward_list : List[Dict]) -> None:
		return



class FictitiousSelfPlayManager(PolicyMappingManager):
	def __init__(self, 
		agent_force_mapping_fn : Callable[[str], str],
		num_subpolicies : int,
		wolf_fn : Optional[Callable[[float], float]] = None,
	):
		""" FictitiousSelfPlayManager
		This class enables plug-and-play style self-play training scheme.
		Agents are divided into two teams "blufor" and "redfor" to learn competitively,
		while each policy may have multiple "subpolicies" to represent various strategies.
		Also known as "Fictitious Self-Play", which helps faster convergence to Nash equilibrium.
		
		Args:
		    agent_force_mapping_fn (Callable[[str], str]): Function to determine which force an agent belongs to. Must return "blufor" or "redfor".
		    num_subpolicies (int): Number of subpolicies
		    wolf_fn (Optional[Callable[[float], float]], optional): Function to scale learning rate based on "Win-or-Learn-Fast" principle.
		"""
		self.agent_force_mapping_fn = agent_force_mapping_fn
		self.num_subpolicies = num_subpolicies
		self.episode_policy_mapping : Dict[int,Tuple[int,int]] = {}
		self.episode_agent_mapping : Dict[int,Dict[str,str]] = {}
		self.match_results = [[[] for ri in range(self.num_subpolicies)] for bi in range(self.num_subpolicies)]
		self.wolf_fn = wolf_fn


	def policy_selection(self) -> Dict[str, int]:
		""" Policy selection
		Returns subpolicy ids for each team
		(i.e., same team agents have same subpolicy through an episode)
		Can be overriden to do something more complicated (e.g. Priorized FSP),
		as long as returning dictionary with "blufor" and "redfor" keys.
		
		Returns:
		    Dict[str, int]: Subpolicy ids for each team
		"""
		return {
			"blufor": np.random.randint(self.num_subpolicies), 
			"redfor": np.random.randint(self.num_subpolicies)
		}


	def subpolicy_name(self, policy_id : str, index : int) -> str:
		""" Subpolicy naming rule (Necessary for rllib communication)
		
		Args:
		    policy_id (str): Policy id
		    index (int): Subpolicy id
		
		Returns:
		    str: Subpolicy name
		"""
		return policy_id + '_' + str(index)


	def get_policy_mapping(self, agent_id : str, policy_id : str, episode : Episode) -> str:
		""" Maps policy to subpolicy
		
		Args:
		    agent_id (str): agent id (Used to determine policy_id)
		    policy_id (str): policy id
		    episode (Episode): episode instance
		
		Returns:
		    str: subpolicy id
		"""

		if episode.episode_id not in self.episode_policy_mapping:
			self.episode_policy_mapping[episode.episode_id] = self.policy_selection()
			self.episode_agent_mapping[episode.episode_id] = {}

		force = self.agent_force_mapping_fn(agent_id)
		assert force == "blufor" or force == "redfor", "agent_force_mapping_fn must return \"blufor\" or \"redfor\"."
		self.episode_agent_mapping[episode.episode_id][agent_id] = force
		return self.subpolicy_name(policy_id, self.episode_policy_mapping[episode.episode_id][force])


	def get_policy_mapping_list(self, policy_id : str) -> List[str]:
		""" Returns all possible subpolicy mappings from a policy
		
		Args:
		    policy_id (str): policy id
		
		Returns:
		    List[str]: All possible subpolicies
		"""
		return [self.subpolicy_name(policy_id, i) for i in range(self.num_subpolicies)]


	def update_policy_configuration(self, trainer : Trainer, reward_list : List[Dict]) -> None:
		""" Update policy configuration
		Calls user-defined configuration function after computing blufor vs redfor subpolicy reward matrix.
		
		Args:
		    trainer (Trainer): Trainer instance
		    reward_list (List[Dict]): Reward list
		"""
		cum_rewards, current_match_results, policy_force_mapping = self.compute_match_results(reward_list, 
			ma_length = len(reward_list) // (self.num_subpolicies ** 2))
		
		if self.wolf_fn is not None:
			for policy_id in policy_force_mapping:
				forces = list(set(policy_force_mapping[policy_id]))
				if len(forces) == 1:
					policy = trainer.get_policy(policy_id)
					scale = self.wolf_fn(np.mean(cum_rewards[forces[0]]))

					if isinstance(policy._lr_schedule, ScheduleScaler):
						policy._lr_schedule = ScheduleScaler(policy._lr_schedule.schedule, scale)
					else:
						policy._lr_schedule = ScheduleScaler(policy._lr_schedule, scale)
				else:
					print('Policy %s was not configured by wolf_fn because it was used in both blufor and redfor.' % policy_id)


	def compute_match_results(self, reward_list : List[Dict], ma_length : int = 1000) -> Tuple[Dict[str,List[float]], np.ndarray, Dict[str,List[str]]]:
		""" Compute blufor vs redfor subpolicy reward matrix
		Result of this function can be used to determine which subpolicy to utilize, which subpolicy to train more,
		or which subpolicy to choose as an opponent.
		Returns None if number of samples are insufficient (Specific subpolicy combination did not appear)
		
		Args:
		    reward_list (List[Dict]): Reward list
		    ma_length (int, optional): Maximum size how old episode to be considered for average reward computation.
		
		Returns:
		    Tuple[Dict[str, List[float]], np.ndarray, Dict[str, List[str]]]: 
		    	Reward list for each force
		    	blufor vs redfor subpolicy reward matrix
		    	Policy-force mapping list
		"""

		# Collect performance results of each subpolicy from trajectory
		cum_reward_list_all = {"blufor": [], "redfor": []}
		policy_force_mapping = {}
		for episode_id, reward_dict in reward_list.items():
			cum_reward_list = {"blufor": [], "redfor": []}
			blufor_subpolicy_index = self.episode_policy_mapping[episode_id]["blufor"]
			redfor_subpolicy_index = self.episode_policy_mapping[episode_id]["redfor"]

			for (agent_id, subpolicy_id), reward in reward_dict.items():
				force = self.episode_agent_mapping[episode_id][agent_id]
				if force not in cum_reward_list:
					print("Unknown force %s in FictitiousSelfPlayManager" % self.episode_agent_mapping[episode_id][agent_id])

				cum_reward_list[force].append(reward)

				if subpolicy_id not in policy_force_mapping:
					policy_force_mapping[subpolicy_id] = []
				policy_force_mapping[subpolicy_id].append(force)

			self.match_results[blufor_subpolicy_index][redfor_subpolicy_index].append(
				np.float32([np.mean(cum_reward_list["blufor"]), np.mean(cum_reward_list["redfor"])]))
			for force in ["blufor", "redfor"]:
				cum_reward_list_all[force].append(np.mean(cum_reward_list[force]))

		# Compute moving average
		current_match_results = np.zeros((self.num_subpolicies, self.num_subpolicies, 2), dtype=np.float32)

		for bi in range(self.num_subpolicies):
			for ri in range(self.num_subpolicies):
				if len(self.match_results[bi][ri]) == 0:
					print('Match result computation failed due to insufficient samples.')
					return cum_reward_list_all, None, policy_force_mapping
		
				if len(self.match_results[bi][ri]) <= ma_length:
					current_match_results[bi][ri] = np.mean(self.match_results[bi][ri], axis=0)
				else:
					current_match_results[bi][ri] = np.mean(self.match_results[bi][ri][-ma_length:], axis=0)

		return cum_reward_list_all, current_match_results, policy_force_mapping



class SelfPlayManager(FictitiousSelfPlayManager):
	""" Specialized case of FictitiousSelfPlayManager where number of subpolicies is one
	"""
	
	def __init__(self, 
		agent_force_mapping_fn : Callable[[str], str],
		wolf_fn : Optional[Callable[[float,float], float]] = None,
	):
		super().__init__(agent_force_mapping_fn, 1, wolf_fn)


	def subpolicy_name(self, policy_id : str, index : int) -> str:
		return policy_id
