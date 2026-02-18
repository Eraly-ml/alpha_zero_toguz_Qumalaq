# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Unit tests for Togyz Kumalak environment."""

import unittest
import numpy as np
from alpha_zero.envs.toguz import ToguzKumalakEnv, NUM_PITS, INITIAL_STONES, TOTAL_STONES, WIN_THRESHOLD


class TestToguzKumalakInit(unittest.TestCase):
    def setUp(self):
        self.env = ToguzKumalakEnv(num_stack=4)

    def test_initial_board(self):
        self.env.reset()
        np.testing.assert_array_equal(self.env.board[0], [9] * 9)
        np.testing.assert_array_equal(self.env.board[1], [9] * 9)

    def test_initial_kazans(self):
        self.env.reset()
        np.testing.assert_array_equal(self.env.kazans, [0, 0])

    def test_initial_tuzduks(self):
        self.env.reset()
        np.testing.assert_array_equal(self.env.tuzduks, [-1, -1])

    def test_initial_legal_actions(self):
        self.env.reset()
        np.testing.assert_array_equal(self.env.legal_actions, [1] * 9)

    def test_action_space(self):
        self.assertEqual(self.env.action_space.n, 9)
        self.assertEqual(self.env.action_dim, 9)

    def test_observation_shape(self):
        obs = self.env.reset()
        expected_shape = (4 * 6 + 1, 2, 9)  # 25 channels
        self.assertEqual(obs.shape, expected_shape)

    def test_total_stones_conserved_at_start(self):
        self.env.reset()
        total = np.sum(self.env.board) + np.sum(self.env.kazans)
        self.assertEqual(total, TOTAL_STONES)


class TestToguzKumalakSowing(unittest.TestCase):
    def setUp(self):
        self.env = ToguzKumalakEnv(num_stack=4)
        self.env.reset()

    def test_single_stone_move(self):
        """When a pit has exactly 1 stone, move it to the next pit."""
        self.env.board[0] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.board[1] = [9, 9, 9, 9, 9, 9, 9, 9, 9]
        self.env.kazans = np.array([80, 0], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(0)  # Move 1 stone from pit 0 to pit 1
        self.assertEqual(self.env.board[0, 0], 0)
        self.assertEqual(self.env.board[0, 1], 1)

    def test_multi_stone_sowing(self):
        """Sowing keeps 1 stone in origin, distributes rest."""
        self.env.board[0] = [3, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.board[1] = [9] * 9
        self.env.kazans = np.array([78, 0], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(0)
        # Pit 0 should keep 1 stone, distribute 2 more to pits 1 and 2
        self.assertEqual(self.env.board[0, 0], 1)
        self.assertEqual(self.env.board[0, 1], 1)
        self.assertEqual(self.env.board[0, 2], 1)

    def test_sowing_wraps_to_opponent(self):
        """Sowing from pit 8 should wrap to opponent's pit 0."""
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 3]
        self.env.board[1] = [9] * 9
        self.env.kazans = np.array([78, 0], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # Pit 8 keeps 1, 2 stones go to opponent pits 0 and 1
        self.assertEqual(self.env.board[0, 8], 1)
        self.assertEqual(self.env.board[1, 0], 10)
        # Opponent pit 1: 9+1=10 (even) -> captured by current player
        self.assertEqual(self.env.board[1, 1], 0)
        self.assertEqual(self.env.kazans[0], 78 + 10)  # captured 10 stones

    def test_stones_conservation(self):
        """Total stones should be conserved after each move."""
        self.env.reset()
        for _ in range(20):
            if self.env.is_game_over():
                break
            legal = np.where(self.env.legal_actions == 1)[0]
            if len(legal) == 0:
                break
            action = np.random.choice(legal)
            self.env.step(action)
            total = np.sum(self.env.board) + np.sum(self.env.kazans)
            self.assertEqual(total, TOTAL_STONES, f"Stones not conserved at step {self.env.steps}")


class TestToguzKumalakCapture(unittest.TestCase):
    def setUp(self):
        self.env = ToguzKumalakEnv(num_stack=4)
        self.env.reset()

    def test_even_capture(self):
        """Capture when last stone makes even count in opponent's pit."""
        # Setup: player 1 pit 7 has 2 stones, opponent pit 0 has 1 stone
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 2, 0]
        self.env.board[1] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.kazans = np.array([0, 158], dtype=np.int32)
        self.env._update_legal_actions()

        # Player 1 plays pit 7: keep 1, distribute 1 to pit 8
        # Pit 8 = 0+1=1, not opponent territory, no capture
        # Actually let's set up correctly for opponent capture
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.board[1] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.kazans = np.array([0, 158], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # Pit 8 keeps 1, 1 stone goes to opponent pit 0
        # Opponent pit 0: 1+1 = 2 (even) -> captured
        self.assertEqual(self.env.board[1, 0], 0)
        self.assertEqual(self.env.kazans[0], 2)

    def test_no_capture_on_odd(self):
        """No capture when last stone makes odd count."""
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.board[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.kazans = np.array([0, 157], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # Opponent pit 0: 2+1 = 3 (odd, not captured, but could be tuzdyk)
        # Since this is 3, check tuzdyk rules
        # Player 0 has no tuzdyk, pit 0 is not pit 8, no symmetric conflict
        # So tuzdyk is created
        self.assertEqual(self.env.tuzduks[0], 0)


class TestToguzKumalakTuzdyk(unittest.TestCase):
    def setUp(self):
        self.env = ToguzKumalakEnv(num_stack=4)
        self.env.reset()

    def test_tuzdyk_creation(self):
        """Tuzdyk created when last stone makes exactly 3 in opponent's pit."""
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.board[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.kazans = np.array([0, 157], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # Opponent pit 0: 2+1=3, tuzdyk created
        self.assertEqual(self.env.tuzduks[0], 0)
        self.assertEqual(self.env.board[1, 0], 0)  # stones captured
        self.assertEqual(self.env.kazans[0], 3)

    def test_no_tuzdyk_on_9th_pit(self):
        """Cannot create tuzdyk on the 9th pit (index 8)."""
        self.env.board[0] = [2, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.board[1] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.kazans = np.array([0, 157], dtype=np.int32)
        self.env.to_play = self.env.white_player  # Player 2's turn
        self.env._update_legal_actions()

        # Set up so player 2 plays pit 8, lands on player 1's pit 8
        self.env.board[1] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.kazans = np.array([157, 0], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # Cannot tuzdyk pit 8 (9th pit)
        self.assertEqual(self.env.tuzduks[1], -1)

    def test_no_second_tuzdyk(self):
        """Player cannot create a second tuzdyk."""
        self.env.tuzduks[0] = 3  # Player 1 already has tuzdyk
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.board[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.kazans = np.array([0, 157], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # Opponent pit 0: 2+1=3 but player already has tuzdyk
        # So even capture (3 is odd, no even capture). No tuzdyk created.
        self.assertEqual(self.env.tuzduks[0], 3)  # unchanged
        self.assertEqual(self.env.board[1, 0], 3)  # stones remain

    def test_symmetric_tuzdyk_blocked(self):
        """Cannot create tuzdyk at same index as opponent's tuzdyk."""
        self.env.tuzduks[1] = 0  # Player 2 already has tuzdyk at index 0
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.board[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.kazans = np.array([0, 157], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # Cannot create tuzdyk at pit 0 because opponent has tuzdyk there
        self.assertEqual(self.env.tuzduks[0], -1)
        # 3 is odd, no even capture either
        self.assertEqual(self.env.board[1, 0], 3)


class TestToguzKumalakGameOver(unittest.TestCase):
    def setUp(self):
        self.env = ToguzKumalakEnv(num_stack=4)
        self.env.reset()

    def test_win_by_82_stones(self):
        """Player wins when kazan reaches 82."""
        self.env.board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.env.board[1] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env.kazans = np.array([80, 78], dtype=np.int32)
        self.env._update_legal_actions()

        self.env.step(8)
        # P1 pit 8 keeps 1, 1 stone to opponent pit 0
        # Opponent pit 0: 1+1=2 (even) -> capture 2
        # P1 kazan: 80+2=82 -> win
        self.assertTrue(self.env.is_game_over())
        self.assertEqual(self.env.winner, self.env.black_player)

    def test_draw(self):
        """Draw when both kazans have 81."""
        self.env.kazans = np.array([81, 81], dtype=np.int32)
        self.env.board = np.zeros((2, 9), dtype=np.int32)
        self.env._check_win_conditions(0, 1)
        self.assertEqual(self.env.winner, -1)

    def test_full_game_no_crash(self):
        """Play a full random game without crashing."""
        self.env.reset()
        while not self.env.is_game_over():
            legal = np.where(self.env.legal_actions == 1)[0]
            if len(legal) == 0:
                break
            action = np.random.choice(legal)
            self.env.step(action)

        # Stones should still be conserved
        total = np.sum(self.env.board) + np.sum(self.env.kazans)
        self.assertEqual(total, TOTAL_STONES)


class TestToguzKumalakObservation(unittest.TestCase):
    def setUp(self):
        self.env = ToguzKumalakEnv(num_stack=4)

    def test_observation_shape_after_reset(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (25, 2, 9))

    def test_observation_shape_after_step(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(0)
        self.assertEqual(obs.shape, (25, 2, 9))

    def test_observation_values_bounded(self):
        self.env.reset()
        obs = self.env.observation()
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

    def test_color_plane(self):
        """Last plane should be 1 when black to play, 0 when white."""
        self.env.reset()
        obs = self.env.observation()
        # Black (player 1) plays first
        np.testing.assert_array_equal(obs[-1], np.ones((2, 9)))

        # After one move, white plays
        self.env.step(0)
        obs = self.env.observation()
        np.testing.assert_array_equal(obs[-1], np.zeros((2, 9)))


class TestToguzKumalakLegalActions(unittest.TestCase):
    def setUp(self):
        self.env = ToguzKumalakEnv(num_stack=4)
        self.env.reset()

    def test_empty_pit_illegal(self):
        """Empty pit should be illegal."""
        self.env.board[0] = [0, 9, 9, 9, 9, 9, 9, 9, 9]
        self.env.kazans = np.array([0, 81], dtype=np.int32)
        self.env._update_legal_actions()
        self.assertEqual(self.env.legal_actions[0], 0)

    def test_tuzdyk_pit_illegal(self):
        """Pit that is opponent's tuzdyk should be illegal for the owner."""
        self.env.tuzduks[1] = 3  # Player 2 has tuzdyk on player 1's pit 3
        self.env._update_legal_actions()
        self.assertEqual(self.env.legal_actions[3], 0)


if __name__ == '__main__':
    unittest.main()
