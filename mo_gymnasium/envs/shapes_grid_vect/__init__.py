from gymnasium.envs.registration import register


register(
    id="shapes-grid-vect-v0",
    entry_point="mo_gymnasium.envs.shapes_grid_vect.shapes_grid_vect:ShapesGridVect",
    max_episode_steps=200,
)
