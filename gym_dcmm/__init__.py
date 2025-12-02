from gymnasium.envs.registration import register

register(
    id="gym_dcmm/DcmmVecWorld-v0",
    entry_point="gym_dcmm.envs.stage1.DcmmVecEnvStage1:DcmmVecEnvStage1",
    max_episode_steps=125,
)

register(
    id="gym_dcmm/DcmmVecWorldCatch-v0",
    entry_point="gym_dcmm.envs.stage2.DcmmVecEnvStage2:DcmmVecEnvStage2",
    max_episode_steps=125,
)