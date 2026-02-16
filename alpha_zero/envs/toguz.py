# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Togyz Kumalak (Тоғыз Құмалақ) environment for AlphaZero.

Togyz Kumalak is a traditional Central Asian mancala game with:
- 2 players, each with 9 pits (otau) and 1 store (kazan)
- 162 total stones (9 per pit initially)
- Counter-clockwise sowing
- Even-count capture from opponent's pits
- Tuzdyk (tuz) mechanism: permanent capture of opponent's pit
- Win condition: first to accumulate 82+ stones
"""
from typing import Tuple
from collections import deque
from copy import copy
import sys
import os

import numpy as np
import gym
from gym.spaces import Box, Discrete

from alpha_zero.envs.base import BoardGameEnv


# Constants
NUM_PITS = 9
INITIAL_STONES = 9
TOTAL_STONES = NUM_PITS * INITIAL_STONES * 2  # 162
WIN_THRESHOLD = TOTAL_STONES // 2 + 1  # 82
NORMALIZATION_FACTOR = float(TOTAL_STONES)

# Player indices (into the board/kazans/tuzduks arrays)
PLAYER_1 = 0  # "Black" - moves first
PLAYER_2 = 1  # "White"


class ToguzKumalakEnv(BoardGameEnv):
    """Togyz Kumalak Environment with OpenAI Gym API.

    Board layout (counter-clockwise direction):
        Player 2 pits:  [8] [7] [6] [5] [4] [3] [2] [1] [0]   <- P2 Kazan
        Player 1 pits:  [0] [1] [2] [3] [4] [5] [6] [7] [8]   -> P1 Kazan

    Sowing goes: P1 pits left-to-right, then P2 pits left-to-right (which is
    right-to-left visually), forming a counter-clockwise loop.
    """

    def __init__(self, num_stack: int = 4) -> None:
        # Initialize parent with board_size=9 (we'll override board shape)
        super().__init__(
            id='Togyz Kumalak',
            board_size=NUM_PITS,
            num_stack=num_stack,
            black_player_id=1,
            white_player_id=2,
            has_pass_move=False,
            has_resign_move=False,
        )

        # Override action space: 9 actions (choose pit 0-8)
        self.action_dim = NUM_PITS
        self.action_space = Discrete(NUM_PITS)

        # Override board: 2 rows x 9 columns (row 0 = player 1, row 1 = player 2)
        self.board = np.full((2, NUM_PITS), INITIAL_STONES, dtype=np.int32)

        # Kazans (stores) for each player
        self.kazans = np.zeros(2, dtype=np.int32)

        # Tuzdyk indices: -1 means no tuzdyk, otherwise index 0-8 of opponent's pit
        self.tuzduks = np.array([-1, -1], dtype=np.int32)

        # Override legal actions
        self.legal_actions = np.ones(NUM_PITS, dtype=np.int8)

        # Override observation space: (num_stack * 6 + 1, 2, 9)
        num_channels = num_stack * 6 + 1
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(num_channels, 2, NUM_PITS),
            dtype=np.float32,
        )

        # Board history for observation stacking
        self.board_history = self._get_empty_history()

        # Maximum steps to prevent infinite games
        self.max_steps = 500

    def reset(self, **kwargs) -> np.ndarray:
        """Reset game to initial state."""
        # Don't call super().reset() as it resets to NxN board
        self.board = np.full((2, NUM_PITS), INITIAL_STONES, dtype=np.int32)
        self.kazans = np.zeros(2, dtype=np.int32)
        self.tuzduks = np.array([-1, -1], dtype=np.int32)

        self.legal_actions = np.ones(NUM_PITS, dtype=np.int8)
        self.to_play = self.black_player
        self.steps = 0
        self.winner = None
        self.last_player = None
        self.last_move = None

        self.board_history = self._get_empty_history()
        del self.history[:]

        return self.observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Play one move.

        Args:
            action: pit index (0-8) to play from current player's side.

        Returns:
            observation, reward, done, info
        """
        if self.is_game_over():
            raise RuntimeError('Game is over, call reset before using step method.')
        if not 0 <= int(action) <= NUM_PITS - 1:
            raise ValueError(f'Invalid action {action}, must be 0-{NUM_PITS - 1}.')
        if self.legal_actions[int(action)] != 1:
            raise ValueError(f'Illegal action {action}.')

        self.last_move = copy(int(action))
        self.last_player = copy(self.to_play)
        self.steps += 1

        self.add_to_history(self.last_player, self.last_move)

        # Get current player index (0 or 1)
        player = self._player_index(self.to_play)
        opponent = 1 - player

        # Execute the move
        self._sow_stones(player, opponent, int(action))

        # Save board state to history
        self.board_history.appendleft(self._snapshot())

        # Check win conditions
        reward = 0.0
        self._check_win_conditions(player, opponent)

        if self.winner is not None:
            if self.winner == self.to_play:
                reward = 1.0
            elif self.winner == -1:
                reward = 0.0  # draw
            else:
                reward = -1.0

        done = self.is_game_over()

        # Switch player
        self.to_play = self.opponent_player

        # Update legal actions for next player
        self._update_legal_actions()

        # Check if next player has no legal moves
        if not done and np.sum(self.legal_actions) == 0:
            next_player = self._player_index(self.to_play)
            next_opponent = 1 - next_player
            # All remaining opponent stones go to opponent's kazan
            self.kazans[next_player] += np.sum(self.board[next_player])
            self.board[next_player] = 0
            # Also give opponent their remaining stones
            self.kazans[next_opponent] += np.sum(self.board[next_opponent])
            self.board[next_opponent] = 0
            self._determine_winner_by_kazans()
            done = True

        return self.observation(), reward, done, {}

    def _sow_stones(self, player: int, opponent: int, pit: int) -> None:
        """Execute stone sowing, capture, and tuzdyk logic."""
        stones = self.board[player, pit]
        self.board[player, pit] = 0

        if stones == 0:
            return

        # Special case: if only 1 stone, move it to the next pit
        if stones == 1:
            # Next pit in counter-clockwise order
            next_pos = self._next_position(player, pit)
            row, col = next_pos
            self.board[row, col] += 1

            # Check for tuzdyk and capture only in opponent's pits
            if row == opponent:
                self._check_tuzdyk_and_capture(player, opponent, col)
            return

        # General case: distribute stones one by one
        # Keep one stone in the origin pit
        self.board[player, pit] = 1
        stones -= 1

        current_row, current_col = player, pit
        for i in range(stones):
            current_row, current_col = self._next_position(current_row, current_col)
            self.board[current_row, current_col] += 1

        # After sowing, check if last stone landed in opponent's pit
        if current_row == opponent:
            self._check_tuzdyk_and_capture(player, opponent, current_col)

    def _check_tuzdyk_and_capture(self, player: int, opponent: int, pit: int) -> None:
        """Check and apply tuzdyk and capture rules for an opponent's pit."""
        count = self.board[opponent, pit]

        # Tuzdyk: if count is exactly 3
        if count == 3:
            if self._can_create_tuzdyk(player, opponent, pit):
                self.tuzduks[player] = pit
                self.kazans[player] += self.board[opponent, pit]
                self.board[opponent, pit] = 0
                return

        # Capture: if count is even
        if count % 2 == 0 and count > 0:
            self.kazans[player] += count
            self.board[opponent, pit] = 0

    def _can_create_tuzdyk(self, player: int, opponent: int, pit: int) -> bool:
        """Check if player can create a tuzdyk at opponent's pit."""
        # Player already has a tuzdyk
        if self.tuzduks[player] != -1:
            return False

        # Cannot make the 9th pit (index 8) a tuzdyk
        if pit == NUM_PITS - 1:
            return False

        # Symmetry rule: both players cannot have tuzdyk at the same index
        if self.tuzduks[opponent] == pit:
            return False

        return True

    def _next_position(self, row: int, col: int) -> Tuple[int, int]:
        """Get next position in counter-clockwise sowing order.

        Order: P1[0]->P1[1]->...->P1[8]->P2[0]->P2[1]->...->P2[8]->P1[0]->...
        """
        if col < NUM_PITS - 1:
            return (row, col + 1)
        else:
            # Wrap to the other player's side
            return (1 - row, 0)

    def _check_win_conditions(self, player: int, opponent: int) -> None:
        """Check if the game has been won."""
        if self.kazans[player] >= WIN_THRESHOLD:
            self.winner = self._player_id(player)
        elif self.kazans[opponent] >= WIN_THRESHOLD:
            self.winner = self._player_id(opponent)
        elif self.kazans[0] == TOTAL_STONES // 2 and self.kazans[1] == TOTAL_STONES // 2:
            self.winner = -1  # Draw
        elif self.steps >= self.max_steps:
            self._determine_winner_by_kazans()

    def _determine_winner_by_kazans(self) -> None:
        """Determine winner based on kazan counts."""
        if self.kazans[0] > self.kazans[1]:
            self.winner = self.black_player
        elif self.kazans[1] > self.kazans[0]:
            self.winner = self.white_player
        else:
            self.winner = -1  # Draw

    def _update_legal_actions(self) -> None:
        """Update legal actions for the current player."""
        player = self._player_index(self.to_play)
        opponent = 1 - player
        self.legal_actions = np.zeros(NUM_PITS, dtype=np.int8)

        for i in range(NUM_PITS):
            # A pit is legal if it has stones and is not the opponent's tuzdyk on our side
            if self.board[player, i] > 0:
                # Check if this pit is captured as opponent's tuzdyk
                if self.tuzduks[opponent] != i:
                    self.legal_actions[i] = 1

    def observation(self) -> np.ndarray:
        """Create observation tensor with shape (C, 2, 9).

        Feature planes per history step (6 planes):
            0: Current player's pit counts (normalized)
            1: Opponent's pit counts (normalized)
            2: Current player's kazan (broadcast, normalized)
            3: Opponent's kazan (broadcast, normalized)
            4: Current player's tuzdyk mask (binary on opponent's pits)
            5: Opponent's tuzdyk mask (binary on current player's pits)
        Plus 1 color-to-play plane.
        """
        player = self._player_index(self.to_play)
        opponent = 1 - player

        planes = []

        for snapshot in self.board_history:
            board_snap, kazans_snap, tuzduks_snap = snapshot

            # Create 6 feature planes, each of shape (2, 9)
            # Plane 0: current player's pits
            p0 = np.zeros((2, NUM_PITS), dtype=np.float32)
            p0[0] = board_snap[player] / NORMALIZATION_FACTOR
            p0[1] = 0  # opponent row is zero in this plane

            # Plane 1: opponent's pits
            p1 = np.zeros((2, NUM_PITS), dtype=np.float32)
            p1[0] = 0
            p1[1] = board_snap[opponent] / NORMALIZATION_FACTOR

            # Plane 2: current player kazan (broadcast)
            p2 = np.full((2, NUM_PITS), kazans_snap[player] / NORMALIZATION_FACTOR, dtype=np.float32)

            # Plane 3: opponent kazan (broadcast)
            p3 = np.full((2, NUM_PITS), kazans_snap[opponent] / NORMALIZATION_FACTOR, dtype=np.float32)

            # Plane 4: current player's tuzdyk (on opponent's pits)
            p4 = np.zeros((2, NUM_PITS), dtype=np.float32)
            if tuzduks_snap[player] >= 0:
                p4[1, tuzduks_snap[player]] = 1.0  # mark on opponent's row

            # Plane 5: opponent's tuzdyk (on current player's pits)
            p5 = np.zeros((2, NUM_PITS), dtype=np.float32)
            if tuzduks_snap[opponent] >= 0:
                p5[0, tuzduks_snap[opponent]] = 1.0  # mark on current player's row

            planes.extend([p0, p1, p2, p3, p4, p5])

        # Color to play plane
        color_plane = np.zeros((2, NUM_PITS), dtype=np.float32)
        if self.to_play == self.black_player:
            color_plane[:] = 1.0

        planes.append(color_plane)

        return np.array(planes, dtype=np.float32)

    def _snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a snapshot of current board state."""
        return (
            np.copy(self.board),
            np.copy(self.kazans),
            np.copy(self.tuzduks),
        )

    def _get_empty_history(self) -> deque:
        """Return empty history queue filled with zero states."""
        empty_snap = (
            np.zeros((2, NUM_PITS), dtype=np.int32),
            np.zeros(2, dtype=np.int32),
            np.array([-1, -1], dtype=np.int32),
        )
        return deque(
            [empty_snap] * self.num_stack,
            maxlen=self.num_stack,
        )

    def is_game_over(self) -> bool:
        if self.winner is not None:
            return True
        if self.steps >= self.max_steps:
            return True
        return False

    def is_board_full(self) -> bool:
        """Not applicable for Togyz Kumalak."""
        return False

    def _player_index(self, player_id: int) -> int:
        """Convert player ID (1 or 2) to array index (0 or 1)."""
        if player_id == self.black_player:
            return PLAYER_1
        return PLAYER_2

    def _player_id(self, index: int) -> int:
        """Convert array index (0 or 1) to player ID (1 or 2)."""
        if index == PLAYER_1:
            return self.black_player
        return self.white_player

    def get_result_string(self) -> str:
        if not self.is_game_over():
            return ''
        if self.winner == self.black_player:
            return f'B+{self.kazans[0]}-{self.kazans[1]}'
        elif self.winner == self.white_player:
            return f'W+{self.kazans[1]}-{self.kazans[0]}'
        else:
            return 'DRAW'

    def render(self, mode='terminal'):
        """Render the board in mancala style."""
        from six import StringIO

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        if mode == 'human':
            if os.name == 'posix':
                os.system('clear')
            else:
                os.system('cls')

        outfile.write(f'\n  {self.id}\n')
        outfile.write(f'  Game over: {"Yes" if self.is_game_over() else "No"}, Result: {self.get_result_string()}\n')
        outfile.write(f'  Steps: {self.steps}, Current player: {"P1" if self.to_play == self.black_player else "P2"}\n')
        outfile.write('\n')

        # Player 2 side (displayed top, reversed order for visual layout)
        p2_pits = self.board[PLAYER_2]
        p1_pits = self.board[PLAYER_1]

        outfile.write(f'  P2 Kazan: {self.kazans[1]:3d}')
        if self.tuzduks[1] >= 0:
            outfile.write(f'  (Tuz: pit {self.tuzduks[1] + 1})')
        outfile.write('\n')

        outfile.write('  ')
        for i in range(NUM_PITS - 1, -1, -1):
            marker = '*' if (self.tuzduks[0] == i) else ' '
            outfile.write(f'[{p2_pits[i]:3d}{marker}]')
        outfile.write('  <- P2\n')

        outfile.write('  ')
        outfile.write('-' * (NUM_PITS * 6))
        outfile.write('\n')

        outfile.write('  ')
        for i in range(NUM_PITS):
            marker = '*' if (self.tuzduks[1] == i) else ' '
            outfile.write(f'[{p1_pits[i]:3d}{marker}]')
        outfile.write('  -> P1\n')

        outfile.write(f'  P1 Kazan: {self.kazans[0]:3d}')
        if self.tuzduks[0] >= 0:
            outfile.write(f'  (Tuz: pit {self.tuzduks[0] + 1})')
        outfile.write('\n')

        # Pit labels
        outfile.write('  ')
        for i in range(NUM_PITS):
            outfile.write(f'  {i + 1:2d}  ')
        outfile.write('\n\n')

        return outfile

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert action to (row, col) for compatibility."""
        if action is None:
            return (-1, -1)
        return (0, int(action))

    def coords_to_action(self, coords: Tuple[int, int]) -> int:
        """Convert (row, col) to action for compatibility."""
        try:
            _, col = coords
            if 0 <= col < NUM_PITS:
                return col
            return None
        except Exception:
            return None

    def to_sgf(self) -> str:
        """SGF not applicable for Togyz Kumalak."""
        return ''

    def close(self):
        """Clean up."""
        self.board_history.clear()
        del self.history[:]
        return super(BoardGameEnv, self).close()

    def fast_clone(self):
        """Fast shallow clone for MCTS simulation — avoids expensive copy.deepcopy.

        Only copies mutable game state; skips gym.Env metadata, spaces, etc.
        """
        clone = object.__new__(self.__class__)
        # Immutable / shared attrs (no copy needed)
        clone.id = self.id
        clone.board_size = self.board_size
        clone.num_stack = self.num_stack
        clone.black_player = self.black_player
        clone.white_player = self.white_player
        clone.has_pass_move = self.has_pass_move
        clone.has_resign_move = self.has_resign_move
        clone.pass_move = self.pass_move
        clone.resign_move = self.resign_move
        clone.action_dim = self.action_dim
        clone.action_space = self.action_space
        clone.observation_space = self.observation_space
        clone.max_steps = self.max_steps
        # Mutable game state (must copy)
        clone.board = np.copy(self.board)
        clone.kazans = np.copy(self.kazans)
        clone.tuzduks = np.copy(self.tuzduks)
        clone.legal_actions = np.copy(self.legal_actions)
        clone.to_play = self.to_play
        clone.steps = self.steps
        clone.winner = self.winner
        clone.last_player = self.last_player
        clone.last_move = self.last_move
        # History — shallow copy deque with copied arrays inside
        clone.board_history = deque(
            [(np.copy(b), np.copy(k), np.copy(t)) for b, k, t in self.board_history],
            maxlen=self.num_stack,
        )
        clone.history = list(self.history)
        return clone
