# KODOKU
## Multi-agent Reinforcement Learning Library
KODOKU is a framework-style wrapper library of rllib (https://github.com/ray-project/ray) to make it easier to implement complicated multi-agent training scheme.

https://user-images.githubusercontent.com/14792604/162190334-2a8acd13-ab65-4a6a-92a8-3ece28cf2cb3.mp4


## Simple Multi-agent Training
Multi-agent trainining is included in ```KODOKUTrainer```.

```
def config_fn():
	return \
		'default', \
		{
			"depth": 2.0,
			"width": 1.0,
			"atk_spawn_line": 1.5,
			"def_spawn_line": 0.5,
			"atk_num" : 3,
			"def_num" : 3,
			"unit_hp" : 1.0,
			"unit_power": 0.1,
			"unit_range": 0.1,
			"unit_speed": 0.05,
			"timelimit": 500,
		}


if __name__ == '__main__':
	trainer = KODOKUTrainer(
		log_dir='./log_dir', 
		env_class=SimpleBattlefieldEnv_Sym,
		train_config=json.load(open('train_config.json')),
		env_config_fn=config_fn,
	)

	trainer.train(10, epoch_callback=callback)
	trainer.evaluate()
```

An example is provided in ```sample/main.py```.

## Self-Play Training
Simple self-play is applied if policy_mapping_fn returns same policy for all agents.
```
def policy_mapping_fn(agent_id, episode, **kwargs):
	return "common"
```

## Fictitious Self-Play Training (FSP)
Self-play can be easily implemented via ```PolicyMappingManager```.

```
trainer = KODOKUTrainer(
	log_dir='./log_dir', 
	env_class=SimpleBattlefieldEnv_Sym,
	train_config=json.load(open('train_config.json')),
	env_config_fn=config_fn,
	# Three subpolicies for each policy
	policy_mapping_manager=SelfPlayManager(lambda agent: "blufor" if agent.startswith("atk") else "redfor", 3),
)
```

An example is provided in ```sample/main_sym.py```.

You can extend FSP to PFSP or other variants by inheriting ```SelfPlayManager``` and override ```policy_selection```.

## Asymmetric Fictitious Self-play Training
FSP can be enforced even when the env is asymmetric.

An example is provided in ```sample/main_asym.py```.

## Win or Learn Fast (WoLF)
WoLF is a technique to stabilize asymmetric competitive multi-agent training by scaling learning rate based on payoff.
In this framework, WoLF is realized via ```lr_schedule```, however you can still use scheduler normally because existing scheduler will be wrapped by ```ScheduleScaler```.

```
trainer = KODOKUTrainer(
	log_dir='./log_dir', 
	env_class=SimpleBattlefieldEnv_Asym,
	# Note: train_config may have lr_schedule as usual.
	train_config=json.load(open('train_config.json')),
	env_config_fn=config_fn,
	policy_mapping_manager=SelfPlayManager(
		agent_force_mapping_fn=lambda agent: "blufor" if agent.startswith("atk") else "redfor",
		wolf_fn=lambda reward: 0.25 if reward > 0 else 1.0)
)
```

An example is provided in ```sample/main_wolf.py```.
