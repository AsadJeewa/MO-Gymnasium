import random
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle
from enum import Enum

class DIFFICULTY(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

# START DEBUG 
difficulty = DIFFICULTY.MEDIUM
# END DEBUG

if(difficulty == DIFFICULTY.EASY):
    MAZE = np.array(
        [
            ["1", " ", "2", " ", "2", " ", " ", "G"],
            [" ", " ", " ", "1", " ", "1", " ", " "],
            [" ", " ", " ", " ", "3", " ", "2", " "],
            ["2", " ", "3", " ", " ", " ", " ", " "],
            [" ", "3", " ", " ", " ", " ", "1", " "],
            [" ", " ", " ", " ", "2", " ", " ", "3"],
            [" ", "1", " ", " ", " ", " ", "2", " "],
            ["S", "3", " ", " ", "3", " ", " ", "1"],
        ]
    )

elif(difficulty == DIFFICULTY.MEDIUM):
    MAZE = np.array([
        ["1", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "G"],
        [" ", " ", " ", " ", " ", "1", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", "2", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "3", " ", " ", " "],
        [" ", " ", " ", "2", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "3", " ", " ", " ", " ", " ", " ", "1", " ", " "],
        [" ", "1", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "3", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", "3", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", "2", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "2", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", "1", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        ["S", " ", " ", " ", " ", " ", " ", "3", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "2", " ", " ", " ", " "],
    ])  


BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)


class FourRoomEasy(gym.Env, EzPickle):
    """
    ## Description
    A discretized version of the gridworld environment introduced in [1]. Here, an agent learns to
    collect shapes with positive reward, while avoid those with negative reward, and then travel to a fixed goal.
    The gridworld is split into four rooms separated by walls with passage-ways.

    References
    ----------
    [1] Barreto, André, et al. "Successor Features for Transfer in Reinforcement Learning." NIPS. 2017.

    ## Observation Space
    The observation contains the 2D position of the agent in the gridworld, plus a binary vector indicating which items were collected.

    ## Action Space
    The action space is discrete with 4 actions: left, up, right, down.

    ## Reward Space
    The reward is a 3-dimensional vector with the following components:
    - +1 if collected a blue square, else 0
    - +1 if collected a green triangle, else 0
    - +1 if collected a red circle, else 0

    ## Starting State
    The agent starts in the lower left of the map.

    ## Episode Termination
    The episode terminates when the agent reaches the goal state, G.

    ## Arguments
    - maze: Array containing the gridworld map. See MAZE for an example.

    ## Credits
    Code adapted from: [Mike Gimelfarb's source](https://github.com/mike-gimelfarb/deep-successor-features-for-transfer/blob/main/source/tasks/gridworld.py).
    """

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3
    ROW_COL = MAZE.shape[0]
    
    GOAL_REWARD = 0.1
    SHAPE_REWARD = 1.0

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, maze=MAZE, log_info=None, specialisation=0):
        """
        Creates a new instance of the shapes environment.

        Parameters
        ----------
        maze : np.ndarray
            an array of string values representing the type of each cell in the environment:
                G indicates a goal state (terminal state)
                _ indicates an initial state (there can be multiple, and one is selected at random
                    at the start of each episode)
                X indicates a barrier
                0, 1, .... 9 indicates the type of shape to be placed in the corresponding cell
                entries containing other characters are treated as regular empty cells
        """
        EzPickle.__init__(self, render_mode, maze)

        self.render_mode = render_mode
        self.window_size = (maze.shape[0] *25)
        self.window = None
        self.clock = None
        self.log_info = log_info
        self.seed = 0
        self.height, self.width = maze.shape
        self.maze = maze
        shape_types = ["1", "2", "3"]
        self.all_shapes = dict(zip(shape_types, range(len(shape_types))))

        self.specialisation = specialisation
        self.goal = None
        self.initial = []
        self.occupied = set()
        self.shape_ids = dict()
        self.shape_type_ids = dict() 
        shapes_by_type = dict()
        shape_types_list = []

        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] == "G":
                    self.goal = (r, c)
                elif maze[r, c] == "S":
                    self.initial.append((r, c))
                elif maze[r, c] == "X":
                    self.occupied.add((r, c))
                elif maze[r, c] in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                    t = int(maze[r, c])
                    self.shape_type_ids[(r, c)] = t
                    shapes_by_type.setdefault(t, []).append((r, c))
        shape_id = 0
        for t in sorted(shapes_by_type.keys()):   # ensures deterministic ordering
            for pos in shapes_by_type[t]:
                self.shape_ids[pos] = shape_id
                shape_id += 1
        
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=np.zeros(2 + len(self.shape_ids)),
            high=len(self.maze) * np.ones(2 + len(self.shape_ids)),
            dtype=np.float32,
        )
        self.reward_space = Box(low=0, high=1, shape=(3,))
        self.reward_dim = 3

    def state_to_array(self, state):
        # converts multiple tuples to an array
        s = [element for tupl in state for element in tupl]
        return np.array(s, dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.seed = seed
        collected = tuple(0 for _ in range(len(self.shape_ids)))
        self.state = (
            random.choice(self.initial),
            collected,
        )
        if self.render_mode == "human":
            self.render()
        return self.state_to_array(self.state), {}

    def step(self, action):
        prev_state = self.state #initially from reset
        (row, col), collected = self.state
        # perform the movement
        if action == FourRoomEasy.LEFT:
            col -= 1
        elif action == FourRoomEasy.UP:
            row -= 1
        elif action == FourRoomEasy.RIGHT:
            col += 1
        elif action == FourRoomEasy.DOWN:
            row += 1
        else:
            raise Exception(f"bad action {action}")

        terminated = False

        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return (
                self.state_to_array(self.state),# obs
                np.zeros(len(self.all_shapes), dtype=np.float32), # reward
                terminated,
                False,
                {},
            )

        # into a blocked cell, cannot move
        pos = (row, col)
        if pos in self.occupied:
            return (
                self.state_to_array(self.state),
                np.zeros(len(self.all_shapes), dtype=np.float32),
                terminated,
                False,
                {},
            )

        # can now move
        self.state = (pos, collected)
        if self.render_mode == "human":
            self.render()

        # into a goal cell
        if pos == self.goal:
            phi = np.ones(len(self.all_shapes), dtype=np.float32)
            terminated = True
            return self.state_to_array(self.state), phi, terminated, False, {}

        # into a shape cell
        if pos in self.shape_ids: #row col
            shape_id = self.shape_ids[pos]
            if collected[shape_id] == 1:
                
                # already collected this flag
                return (
                    self.state_to_array(self.state),
                    np.zeros(len(self.all_shapes), dtype=np.float32),
                    terminated,
                    False,
                    {},
                )
            else:
                # collect the new flag
                collected = list(collected)
                collected[shape_id] = 1
                collected = tuple(collected)
                self.state = (pos, collected)
                phi = self.calc_vect_reward(prev_state, self.state)
                return self.state_to_array(self.state), phi, terminated, False, {}

        # into an empty cell
        return (
            self.state_to_array(self.state),
            np.zeros(len(self.all_shapes), dtype=np.float32),
            terminated,
            False,
            {},
        )

    def get_spec_obs(self):
        pos, collected_all = self.state
        masked = self.get_masked_collected(collected_all)
        return np.expand_dims(self.state_to_array((pos, masked)), axis=0)

    def calc_vect_reward(self, state, next_state):
        pos, _ = next_state
        _, collected = state
        phi = np.zeros(len(self.all_shapes), dtype=np.float32)
        if pos in self.shape_ids:
            if collected[self.shape_ids[pos]] != 1:
                y, x = pos
                shape_index = self.all_shapes[self.maze[y, x]]
                if self.specialisation == 0:
                    phi[shape_index] = 1.0
                else:
                    if int(self.maze[y, x]) == self.specialisation:
                        phi[shape_index] = 1.0
        elif pos == self.goal:
            phi = np.ones(len(self.all_shapes), dtype=np.float32) * self.GOAL_REWARD
        return phi

    def render(self):
        # all shapes
        top_bar_height = 50  # height of the top message area
        # The size of a single grid square in pixels
        pix_square_size = self.window_size // self.height
        maze_offset = top_bar_height 
        
        if self.window is None and self.render_mode is not None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size + top_bar_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size+top_bar_height))
        canvas.fill((255, 255, 255))

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)
        
        pygame.draw.rect(canvas, (200, 200, 200), pygame.Rect(0, 0, self.window_size, top_bar_height))

        if self.log_info is not None:
            img = self.font.render(self.to_obj_string(self.log_info), True, BLACK)
            canvas.blit(img, (10, 10))  # 10 px padding from top-left
        
        img = self.font.render("G", True, BLACK)
        canvas.blit(img, (np.array(self.goal)[::-1] + 0.15) * pix_square_size + np.array([0, maze_offset]))
        img = self.font.render("S", True, BLACK)
        canvas.blit(img, (np.array(self.initial[0])[::-1] + 0.15) * pix_square_size + np.array([0, maze_offset]))

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                (row, col), collected = self.state
                shape_id = self.shape_ids.get((i, j), 0)
                if collected[shape_id] == 1 and self.maze[i, j] != "X":
                    continue

                pos = np.array([j, i])  
                if self.maze[i, j] == "1":
                    pygame.draw.rect(
                        canvas,
                        BLUE,
                        pygame.Rect(
                            pix_square_size * pos + np.array([0, maze_offset]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.maze[i, j] == "X":
                    pygame.draw.rect(
                        canvas,
                        BLACK,
                        pygame.Rect(
                            pix_square_size * pos + np.array([0, maze_offset]) + 1,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.maze[i, j] == "2":
                    pygame.draw.polygon(
                        canvas,
                        GREEN,
                        [
                            (pos + np.array([0.5, 0.0])) * pix_square_size + np.array([0, maze_offset]),
                            (pos + np.array([0.0, 1.0])) * pix_square_size + np.array([0, maze_offset]),
                            (pos + 1.0) * pix_square_size + np.array([0, maze_offset]),
                        ],
                    )
                elif self.maze[i, j] == "3":
                    pygame.draw.circle(
                        canvas,
                        RED,
                        (pos + 0.5) * pix_square_size + np.array([0, maze_offset]),
                        pix_square_size / 2,
                    )
        player_pos = np.array(self.state[0])[::-1]
        pygame.draw.circle(
            canvas,
            (125, 125, 125),
            (player_pos + 0.5) * pix_square_size + np.array([0, maze_offset]),
            pix_square_size / 3,
        )

        # Horizontal lines
        for i in range(self.height + 1):
            y = i * pix_square_size + maze_offset
            width = 3 if i == 0 or i == self.height else 1
            draw_line_dashed(canvas, BLACK, (0, y), (self.window_size, y), width=width)

        # Vertical lines
        for j in range(self.width + 1):
            x = j * pix_square_size
            width = 3 if j == 0 or j == self.width else 1
            draw_line_dashed(canvas, BLACK, (x, maze_offset), (x, self.window_size + maze_offset), width=width)


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def update_specialisation(self,specialisation):
        self.specialisation = specialisation
        return np.expand_dims(self.get_spec_obs(), axis=0) #TODO check

    def get_masked_collected(self, collected_all):
        collected_all = np.array(collected_all, dtype=np.float32)
        shape_types_list = [self.shape_type_ids[pos] for pos, _ in sorted(self.shape_ids.items(), key=lambda kv: kv[1])]
        if self.specialisation == 0:
            return np.array(collected_all)
        active_indices = [i for i, v in enumerate(shape_types_list) if v == self.specialisation]
        mask = np.zeros(len(collected_all), dtype=np.float32)
        mask[active_indices] = 1.0
        return np.array(collected_all*mask)

    def to_obj_string(self, obj_num):
        if obj_num == 0:
            return "blue"
        if obj_num == 1:
            return "green"
        if obj_num == 2:
            return "red"

def draw_line_dashed(surface, color, start_pos, end_pos, width=1, dash_length=3, exclude_corners=True):
    """Code from https://codereview.stackexchange.com/questions/70143/drawing-a-dashed-line-with-pygame."""
    # convert tuples to numpy arrays
    if dash_length < 1:
        pygame.draw.line(surface, color, start_pos, end_pos, width)
    else:
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        # get euclidean distance between start_pos and end_pos
        length = np.linalg.norm(end_pos - start_pos)

        # get amount of pieces that line will be split up in (half of it are amount of dashes)
        dash_amount = int(length / dash_length)

        # x-y-value-pairs of where dashes start (and on next, will end)
        dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()

        return [
            pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n + 1]), width)
            for n in range(int(exclude_corners), dash_amount - int(exclude_corners), 2)
        ]


if __name__ == "__main__":
    import mo_gymnasium as mo_gym

    env = mo_gym.make("four-room-easy-v0", render_mode="human")
    terminated = False
    env.reset()
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
