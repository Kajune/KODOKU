from typing import *
from gym import Env, spaces
import numpy as np
import cv2
from envWrapper import EnvWrapper


class SimpleBattlefieldUnit:
	def __init__(self, 
		pos : np.ndarray, 
		hp : float, 
		power : float,
		range : float, 
		speed : float,
		name : str):
		self.pos = pos
		self.hp = hp
		self.max_hp = hp
		self.power = power
		self.range = range
		self.speed = speed
		self.accel = np.float32([0, 0])
		self.name = name


	def step(self, zoc : float, pos_min : np.ndarray, pos_max : np.ndarray):
		if self.hp <= 0:
			return

		self.pos += self.speed * self.accel * (1 - zoc)
		self.pos = np.clip(self.pos, a_min=pos_min, a_max=pos_max)


	def __str__(self):
		return self.name


class SimpleBattlefieldSimulator:
	def __init__(self, 
		depth : float,
		width : float,
		atk_spawn_line : float,
		def_spawn_line : float,
		def_line : float,
		atk_num : int,
		def_num : int,
		unit_hp : float,
		unit_power : float,
		unit_range : float,
		unit_speed : float,
		timelimit : int,
		**kwargs):

		self.depth = depth
		self.width = width
		self.def_line = def_line
		self.timelimit = timelimit

		self.atk_units = [
			SimpleBattlefieldUnit(np.array([np.random.uniform(atk_spawn_line, depth), np.random.uniform(0, width)]), 
				unit_hp, unit_power, unit_range, unit_speed, 'atk' + str(i)) 
			for i in range(atk_num)
		]

		self.def_units = [
			SimpleBattlefieldUnit(np.array([np.random.uniform(0, def_spawn_line), np.random.uniform(0, width)]), 
				unit_hp, unit_power, unit_range, unit_speed, 'def' + str(i)) 
			for i in range(def_num)
		]

		self.count = 0


	def step(self):
		self.count += 1

		# calc zoc
		atk_pos = np.array([unit.pos for unit in self.atk_units])
		def_pos = np.array([unit.pos for unit in self.def_units])
		atk_power = np.array([unit.power for unit in self.atk_units])
		def_power = np.array([unit.power for unit in self.def_units])
		atk_range = np.array([unit.range for unit in self.atk_units])
		def_range = np.array([unit.range for unit in self.def_units])
		atk_integrity = np.maximum(np.array([unit.hp / unit.max_hp for unit in self.atk_units]), 0)
		def_integrity = np.maximum(np.array([unit.hp / unit.max_hp for unit in self.def_units]), 0)

		dist = np.linalg.norm(atk_pos[np.newaxis,:,:] - def_pos[:,np.newaxis,:], axis=2)
		zoc_atk = np.sqrt(np.maximum(1 - 3 * dist / (4 * atk_range), 0)) * atk_integrity
		zoc_def = np.sqrt(np.maximum(1 - 3 * dist.T / (4 * def_range), 0)) * def_integrity

		# calc damage
		damage_atk = np.sum(zoc_atk * atk_power, axis=1)
		damage_def = np.sum(zoc_def * def_power, axis=1)
		atk_killed = 0
		def_killed = 0
		for ui, unit in enumerate(self.atk_units):
			hp_old = unit.hp
			unit.hp -= damage_def[ui]
			if hp_old > 0 and unit.hp <= 0:
				atk_killed += 1

		for ui, unit in enumerate(self.def_units):
			hp_old = unit.hp
			unit.hp -= damage_atk[ui]
			if hp_old > 0 and unit.hp <= 0:
				def_killed += 1

		# move units
		for ui, unit in enumerate(self.atk_units):
			unit.step(np.sum(zoc_def[ui]), np.float32([0, 0]), np.float32([self.depth, self.width]))
		for ui, unit in enumerate(self.def_units):
			unit.step(np.sum(zoc_atk[ui]), np.float32([0, 0]), np.float32([self.depth, self.width]))

		# event
		events = {}
		events['ATK_KILLED'] = atk_killed
		events['DEF_KILLED'] = def_killed

		atk_mean_pos_old = np.mean(atk_pos[:,0])

		atk_pos = np.array([unit.pos for unit in self.atk_units])
		atk_integrity = np.maximum(np.array([unit.hp / unit.max_hp for unit in self.atk_units]), 0)

		events['ATK_APPROACH'] = (np.mean(atk_pos[:,0]) - atk_mean_pos_old) / (self.depth - self.def_line)

		if np.sum((atk_pos[:,0] <= self.def_line) & (atk_integrity > 0)) >= len(self.atk_units) / 2:
			events['ATK_BREACH'] = 1
		elif np.sum(atk_integrity > 0) <= len(self.atk_units) / 2:
			events['ATK_EXTINCT'] = 1
		elif self.count >= self.timelimit:
			events['TIMELIMIT'] = 1

		return events


class SimpleBattlefieldEnv(EnvWrapper):
	def __init__(self, config_fn : Callable[[], Tuple[str, Dict]]):
		super().__init__(config_fn)
		self.viewer = None


	def log(self) -> Dict:
		return {"events": self.events}


	def initialize_env(self, config : Dict) -> Any:
		self.events = {}
		return SimpleBattlefieldSimulator(**config)


	def get_spaces(self) -> Dict[str, Tuple[spaces.Space, spaces.Space]]:
		obs_space = spaces.Box(low=0, high=1, shape=(3 * (len(self.env.atk_units) + len(self.env.def_units)),), dtype=np.float32)
		act_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

		space_dict = {str(unit): (obs_space, act_space) for unit in self.env.atk_units + self.env.def_units}
		return space_dict


	def get_obs(self) -> Dict[str, Any]:
		def append_obs(obs, unit):
			obs.append(unit.pos[0])
			obs.append(unit.pos[1])
			obs.append(unit.hp / unit.max_hp)
			return obs

		def make_obs(blue_agents, red_agents):
			obs_dict = {}
			for ai, agent in enumerate(blue_agents):
				if agent.hp <= 0:
					continue

				obs = []
				for i in range(len(blue_agents)):
					unit = blue_agents[(ai + i) % len(blue_agents)]
					obs = append_obs(obs, unit)

				for i in range(len(red_agents)):
					unit = red_agents[i]
					obs = append_obs(obs, unit)

				obs_dict[str(agent)] = np.array(obs)
			return obs_dict

		return {**make_obs(self.env.atk_units, self.env.def_units), **make_obs(self.env.def_units, self.env.atk_units)}


	def step(self, action : Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict]:
		for unit in self.env.atk_units + self.env.def_units:
			if str(unit) in action:
				unit.accel = action[str(unit)]

		self.events = self.env.step()

		rewards = {str(unit): 0 for unit in self.env.atk_units + self.env.def_units}
		dones = {str(unit): False for unit in self.env.atk_units + self.env.def_units}
		dones['__all__'] = False

		kill_reward_scale = 0.1
		approach_reward_scale = 1.0
		breach_reward = 1
		extinct_reward = 1
		timelimit_reward = 0.5

		def calc_reward(rewards, unit_list, scale):
			for unit in unit_list:
				rewards[str(unit)] -= kill_reward_scale * self.events['ATK_KILLED'] * scale
				rewards[str(unit)] += kill_reward_scale * self.events['DEF_KILLED'] * scale
				rewards[str(unit)] += approach_reward_scale * self.events['ATK_APPROACH'] * scale

				if 'ATK_BREACH' in self.events:
					rewards[str(unit)] += breach_reward * scale
					dones['__all__'] = True
				if 'ATK_EXTINCT' in self.events:
					rewards[str(unit)] -= extinct_reward * scale
					dones['__all__'] = True
				if 'TIMELIMIT' in self.events:
					rewards[str(unit)] -= timelimit_reward * scale
					dones['__all__'] = True

				if unit.hp <= 0:
					dones[str(unit)] = True
			return rewards

		rewards = calc_reward(rewards, self.env.atk_units, 1.0)
		rewards = calc_reward(rewards, self.env.def_units, -1.0)

		return self.get_obs(), rewards, dones, {}


	def render(self, mode : str = 'human'):
		if mode == 'human':
			img_scale = 256
			height = int(self.env.width * img_scale)
			width = int(self.env.depth * img_scale)

			img = np.ones((height, width, 3), np.uint8) * 255
			img = cv2.line(img, (int(self.env.def_line * img_scale), 0), (int(self.env.def_line * img_scale), height), (0,0,0), 2)
			for unit in self.env.atk_units:
				img = cv2.circle(img, (int(unit.pos[0] * img_scale), int(unit.pos[1] * img_scale)), 
					int(unit.range * img_scale), (int(255 * unit.hp / unit.max_hp),0,0), 1)
			for unit in self.env.def_units:
				img = cv2.circle(img, (int(unit.pos[0] * img_scale), int(unit.pos[1] * img_scale)), 
					int(unit.range * img_scale), (0,0,int(255 * unit.hp / unit.max_hp)), 1)

			cv2.imshow('', img)
			cv2.waitKey(1)

		else:
			raise NotImplementedError()


def config_fn():
	return \
		'default', \
		{
			"depth": 2.0,
			"width": 1.0,
			"atk_spawn_line": 1.5,
			"def_spawn_line": 0.5,
			"def_line": 0.5,
			"atk_num" : 4,
			"def_num" : 3,
			"unit_hp" : 1.0,
			"unit_power": 0.1,
			"unit_range": 0.1,
			"unit_speed": 0.05,
			"timelimit": 500,
		}


if __name__ == '__main__':
	env = SimpleBattlefieldEnv(config_fn)

	for i in range(1):
		obs = env.reset()
		while True:
			print(obs.keys())
			action = {}
			for agent, (obs_space, act_space) in env.get_spaces().items():
				action[agent] = act_space.sample()
			obs, rewards, dones, info = env.step(action)
			print(rewards, dones, info, env.log())

			env.render()

			if dones['__all__']:
				break
