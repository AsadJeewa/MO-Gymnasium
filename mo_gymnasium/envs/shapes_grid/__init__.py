from gymnasium.envs.registration import register


register(
    id="shapes-grid-v0",
    entry_point="mo_gymnasium.envs.shapes_grid.shapes_grid:ShapesGrid",
    max_episode_steps=200,
)
