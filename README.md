# KODOKU
## Multi-agent Self-Play Reinforcement Learning Library
KODOKU is a wrapper library of rllib (https://github.com/ray-project/ray) to make it easier to implement complicated multi-agent training scheme.

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
			"def_line": 0.5,
			"atk_num" : 4,
			"def_num" : 3,
			"unit_hp" : 1.0,
			"unit_power": 0.1,
			"unit_range": 0.1,
			"unit_speed": 0.05,
			"timelimit": 500,
		}

trainer = KODOKUTrainer(
	log_dir='./log_dir', 
	env_class=SimpleBattlefieldEnv,
	train_config=json.load(open('train_config.json')),
	env_config_fn=config_fn,
)

trainer.train(num_epochs=10)
```

An example is provided in

## Self-play Training
Self-play can be easily implemented via ```PolicyMappingManager```.

```
trainer = KODOKUTrainer(
	log_dir='./log_dir', 
	env_class=SimpleBattlefieldEnv,
	train_config=json.load(open('train_config.json')),
	env_config_fn=config_fn,
	policy_mapping_manager=SelfPlayPolicyMappingManager(3), # Three sub-policies per policy
)
```

An example is provided in

## Win or Learn Fast (WoLF)
