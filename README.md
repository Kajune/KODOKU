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
