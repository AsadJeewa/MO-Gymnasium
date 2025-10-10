from gymnasium.envs.registration import register


register(
    id="shapes-grid-v0",
    entry_point="mo_gymnasium.envs.four_room_easy.four_room_easy:FourRoomEasy",
    max_episode_steps=200,
)
