# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Play against the AlphaZero agent on Togyz Kumalak game."""
from absl import flags
import timeit
import os
import sys
import torch

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'num_stack',
    4,
    'Stack N previous states.',
)
flags.DEFINE_integer('num_res_blocks', 6, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer('num_filters', 64, 'Number of filters for the conv2d layers in the neural network.')
flags.DEFINE_integer(
    'num_fc_units',
    128,
    'Number of hidden units in the linear layer of the neural network.',
)
flags.DEFINE_bool('use_se', False, 'Use Squeeze-and-Excitation attention in residual blocks.')

flags.DEFINE_string(
    'black_ckpt',
    './checkpoints/toguz/best_model.ckpt',
    'Load the checkpoint file for black player.',
)
flags.DEFINE_string(
    'white_ckpt',
    './checkpoints/toguz/best_model.ckpt',
    'Load the checkpoint file for white player.',
)

flags.DEFINE_integer('num_simulations', 200, 'Number of iterations per MCTS search.')
flags.DEFINE_integer(
    'num_parallel',
    4,
    'Number of leaves to collect before using the neural network to evaluate the positions during MCTS search, 1 means no parallel search.',
)

flags.DEFINE_float('c_puct_base', 19652, 'Exploration constants balancing priors vs. search values.')
flags.DEFINE_float('c_puct_init', 1.25, 'Exploration constants balancing priors vs. search values.')

flags.DEFINE_bool('human_vs_ai', True, 'Black player is human, default on.')
flags.DEFINE_string('human_color', 'black', 'Human plays as "black" (player 1, moves first) or "white" (player 2).')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

# Initialize flags
FLAGS(sys.argv)

from alpha_zero.envs.toguz import ToguzKumalakEnv
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.pipeline import create_mcts_player, set_seed, disable_auto_grad
from alpha_zero.utils.util import create_logger


def main():
    set_seed(FLAGS.seed)
    logger = create_logger()

    runtime_device = 'cpu'
    if torch.cuda.is_available():
        runtime_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        runtime_device = 'mps'

    eval_env = ToguzKumalakEnv(num_stack=FLAGS.num_stack)

    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    def network_builder():
        return AlphaZeroNet(
            input_shape,
            num_actions,
            FLAGS.num_res_blocks,
            FLAGS.num_filters,
            FLAGS.num_fc_units,
            False,  # Not gomoku
            use_se=FLAGS.use_se,
        )

    def load_checkpoint_for_net(network, ckpt_file, device):
        if ckpt_file and os.path.isfile(ckpt_file):
            loaded_state = torch.load(ckpt_file, map_location=torch.device(device))
            network.load_state_dict(loaded_state['network'])
            logger.info(f'Loaded checkpoint from "{ckpt_file}"')
        else:
            logger.warning(f'Invalid checkpoint file "{ckpt_file}", using random weights.')

    def mcts_player_builder(ckpt_file, device):
        network = network_builder().to(device)
        disable_auto_grad(network)
        load_checkpoint_for_net(network, ckpt_file, device)
        network.eval()

        return create_mcts_player(
            network=network,
            device=device,
            num_simulations=FLAGS.num_simulations,
            num_parallel=FLAGS.num_parallel,
            root_noise=False,
            deterministic=False,
        )

    # Build AI players
    human_is_black = FLAGS.human_color.lower() == 'black'

    if FLAGS.human_vs_ai:
        if human_is_black:
            black_player = 'human'
            white_player = mcts_player_builder(FLAGS.white_ckpt, runtime_device)
        else:
            black_player = mcts_player_builder(FLAGS.black_ckpt, runtime_device)
            white_player = 'human'
    else:
        black_player = mcts_player_builder(FLAGS.black_ckpt, runtime_device)
        white_player = mcts_player_builder(FLAGS.white_ckpt, runtime_device)

    # Stats
    game_count = 0
    human_wins = 0
    ai_wins = 0
    draws = 0

    while True:
        _ = eval_env.reset()
        game_count += 1
        print(f'\n{"="*50}')
        print(f'  Game #{game_count}  |  Human: {human_wins}  AI: {ai_wins}  Draws: {draws}')
        print(f'{"="*50}')

        start = timeit.default_timer()
        while True:
            if eval_env.to_play == eval_env.black_player:
                current_player = black_player
            else:
                current_player = white_player

            if current_player == 'human':
                eval_env.render('human')
                move = None
                while move is None:
                    try:
                        user_input = input('Enter pit number (1-9), or "q" to quit: ').strip()
                        if user_input.lower() == 'q':
                            print(f'\nFinal Score: Human {human_wins} - AI {ai_wins} - Draws {draws}')
                            eval_env.close()
                            return
                        pit = int(user_input) - 1  # Convert 1-indexed to 0-indexed
                        if 0 <= pit <= 8 and eval_env.legal_actions[pit] == 1:
                            move = pit
                        else:
                            legal = [str(i + 1) for i in range(9) if eval_env.legal_actions[i] == 1]
                            print(f'  Invalid move. Legal moves: {", ".join(legal)}')
                    except ValueError:
                        print('  Please enter a number 1-9.')
            else:
                print('  AI is thinking...')
                move, *_ = current_player(eval_env, None, FLAGS.c_puct_base, FLAGS.c_puct_init)
                player_name = 'P1' if eval_env.to_play == eval_env.black_player else 'P2'
                print(f'  AI ({player_name}) plays pit {move + 1}')

            _, _, done, _ = eval_env.step(move)
            eval_env.render('terminal')

            if done:
                break

        duration = timeit.default_timer() - start

        # Determine result
        if eval_env.winner == -1:
            draws += 1
            result_str = 'DRAW'
        elif eval_env.winner == eval_env.black_player:
            if (FLAGS.human_vs_ai and human_is_black):
                human_wins += 1
            else:
                ai_wins += 1
            result_str = f'Player 1 wins! ({eval_env.kazans[0]}-{eval_env.kazans[1]})'
        else:
            if (FLAGS.human_vs_ai and not human_is_black):
                human_wins += 1
            else:
                ai_wins += 1
            result_str = f'Player 2 wins! ({eval_env.kazans[1]}-{eval_env.kazans[0]})'

        print(f'\n  Result: {result_str}')
        print(f'  Game length: {eval_env.steps} moves, {duration:.1f}s')
        print(f'  Score: Human {human_wins} - AI {ai_wins} - Draws {draws}')

        again = input('\n  Play again? (y/n): ').strip().lower()
        if again != 'y':
            break

    eval_env.close()


if __name__ == '__main__':
    main()
