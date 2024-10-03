from gymnasium.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='my_package.envs.grid_world:GridWorldEnv',
)

register(
    id='Cliff-v0',
    entry_point='my_package.envs.cliff:CliffEnv',
)

register(
    id='Ship2D-v0',
    entry_point='my_package.envs.ship:ShipEnv2D',
)

register(
    id='RewardExplorer-v0',
    entry_point='my_package.envs.reward_explorer:RewardExplorerEnv',
    max_episode_steps=500,
)

register(
    id='ShipQuest-v0',
    entry_point='my_package.envs.ship_quest:ShipQuestEnv',
    max_episode_steps=5000,
)