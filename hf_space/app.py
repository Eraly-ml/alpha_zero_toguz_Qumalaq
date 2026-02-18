"""Togyz Kumalak — Play against AlphaZero AI in your browser.

Hugging Face Spaces demo.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import copy
import random
import numpy as np
import torch
import gradio as gr

from alpha_zero.envs.toguz import ToguzKumalakEnv
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.mcts_v2 import (
    Node, DummyNode, expand, backup, best_child,
    add_dirichlet_noise, generate_search_policy,
    parallel_uct_search,
)

# ── Config ──
NUM_STACK = 4
NUM_RES_BLOCKS = 6
NUM_FILTERS = 128
NUM_FC_UNITS = 256
CKPT_PATH = "checkpoints/training_steps_4800.ckpt"
C_PUCT_BASE = 19652.0
C_PUCT_INIT = 1.25

# ── Load model once ──
device = torch.device('cpu')

env_template = ToguzKumalakEnv(num_stack=NUM_STACK)
input_shape = env_template.observation_space.shape
num_actions = env_template.action_space.n

net = AlphaZeroNet(input_shape, num_actions, NUM_RES_BLOCKS, NUM_FILTERS, NUM_FC_UNITS, False)
if os.path.isfile(CKPT_PATH):
    state_dict = torch.load(CKPT_PATH, map_location=device)
    net.load_state_dict(state_dict['network'])
    print(f"Loaded checkpoint: {CKPT_PATH}")
else:
    print("WARNING: No checkpoint found, AI uses random weights!")
net.eval()
for p in net.parameters():
    p.requires_grad = False


@torch.no_grad()
def eval_position(state_np, batched=False):
    if not batched:
        state_np = state_np[None, ...]
    state_t = torch.from_numpy(state_np).float()
    pi_logits, v = net(state_t)
    pi = torch.softmax(pi_logits, dim=-1).numpy()
    v = v.squeeze(-1).numpy().tolist()
    B = state_t.shape[0]
    pi_list = [pi[i] for i in range(B)]
    if not batched:
        return pi_list[0], v[0]
    return pi_list, v


def ai_move(env, num_simulations=50, num_parallel=4):
    """Run MCTS and return the chosen move."""
    move, _, _, _, _ = parallel_uct_search(
        env=env,
        eval_func=eval_position,
        root_node=None,
        c_puct_base=C_PUCT_BASE,
        c_puct_init=C_PUCT_INIT,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=False,
        warm_up=False,
        deterministic=True,
    )
    return move


# ── HTML Board Renderer ──

def render_board(env, message="", last_human=None, last_ai=None,
                 game_over=False, human_is_p1=True):
    board = env.board
    kazans = env.kazans
    tuzduks = env.tuzduks

    if human_is_p1:
        top_player, bot_player = 1, 0
        top_label, bot_label = "AI (Player 2)", "You (Player 1)"
        top_kazan, bot_kazan = kazans[1], kazans[0]
        top_tuz_owner, bot_tuz_owner = 0, 1  # who owns the tuz in this row
    else:
        top_player, bot_player = 0, 1
        top_label, bot_label = "AI (Player 1)", "You (Player 2)"
        top_kazan, bot_kazan = kazans[0], kazans[1]
        top_tuz_owner, bot_tuz_owner = 1, 0

    def pit_html(player_row, pit_i, is_top, last_move, tuz_owner):
        stones = board[player_row, pit_i]
        is_tuz = tuzduks[tuz_owner] == pit_i
        is_last = last_move == pit_i if last_move is not None else False

        border_color = "#e8a838" if is_last else "#555"
        border_w = "3px" if is_last else "1px"
        if is_top:
            bg = "#7c4a4a" if player_row == 1 else "#4a7c59"
        else:
            bg = "#4a7c59" if player_row == 0 else "#7c4a4a"

        tuz_badge = '<span style="position:absolute;top:2px;right:4px;color:#ffcc00;font-size:11px;font-weight:bold;">T</span>' if is_tuz else ''

        return f'''<div style="position:relative;width:62px;height:78px;background:{bg};
            border:{border_w} solid {border_color};border-radius:8px;display:flex;
            flex-direction:column;align-items:center;justify-content:center;margin:2px;">
            {tuz_badge}
            <span style="color:#e0e0e0;font-size:18px;font-weight:bold;">{stones}</span>
            <span style="color:#888;font-size:11px;">pit {pit_i+1}</span>
        </div>'''

    # Top row — reversed display (pit 8 on left, pit 0 on right)
    top_pits = ''.join(pit_html(top_player, 8 - i, True,
                                last_ai if human_is_p1 else last_human,
                                top_tuz_owner) for i in range(9))

    # Bottom row — normal order (pit 0 on left, pit 8 on right)
    bot_pits = ''.join(pit_html(bot_player, i, False,
                                last_human if human_is_p1 else last_ai,
                                bot_tuz_owner) for i in range(9))

    # Legal moves highlight (for bottom row)
    legal = env.legal_actions if not game_over else np.zeros(9, dtype=np.int8)
    is_human_turn = (env.to_play == (env.black_player if human_is_p1 else env.white_player))

    # Message styling
    msg_color = "#e8a838" if "win" in message.lower() or "your turn" in message.lower() else "#e0e0e0"
    if "ai wins" in message.lower():
        msg_color = "#ff6b6b"
    elif "you win" in message.lower():
        msg_color = "#6bff6b"

    html = f'''
    <div style="background:#2b2b2b;border-radius:16px;padding:20px;max-width:700px;margin:0 auto;font-family:sans-serif;">
        <h2 style="text-align:center;color:#e8a838;margin:0 0 5px 0;">&#127922; Тоғыз Құмалақ</h2>
        <p style="text-align:center;color:{msg_color};font-size:16px;margin:5px 0 15px 0;min-height:24px;">{message}</p>

        <div style="display:flex;align-items:center;gap:8px;justify-content:center;">
            <!-- AI Kazan -->
            <div style="width:75px;height:170px;background:{"#6b3d3d" if top_player==1 else "#3d6b4a"};
                border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;
                border:1px solid #555;">
                <span style="color:#e0e0e0;font-size:22px;font-weight:bold;">{top_kazan}</span>
                <span style="color:#888;font-size:10px;">Kazan</span>
            </div>

            <!-- Pits -->
            <div style="display:flex;flex-direction:column;gap:6px;">
                <div style="text-align:center;color:#888;font-size:12px;">{top_label}</div>
                <div style="display:flex;">{top_pits}</div>
                <div style="border-top:1px dashed #555;margin:2px 0;"></div>
                <div style="display:flex;">{bot_pits}</div>
                <div style="text-align:center;color:#888;font-size:12px;">{bot_label}</div>
            </div>

            <!-- Your Kazan -->
            <div style="width:75px;height:170px;background:{"#3d6b4a" if bot_player==0 else "#6b3d3d"};
                border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;
                border:1px solid #555;">
                <span style="color:#e0e0e0;font-size:22px;font-weight:bold;">{bot_kazan}</span>
                <span style="color:#888;font-size:10px;">Kazan</span>
            </div>
        </div>

        <div style="text-align:center;margin-top:12px;color:#888;font-size:12px;">
            Score: You {bot_kazan} — {top_kazan} AI &nbsp;|&nbsp; Need 82 to win
        </div>
    </div>'''
    return html


# ── Game State ──

def create_initial_state():
    env = ToguzKumalakEnv(num_stack=NUM_STACK)
    env.reset()
    return {
        'env': env,
        'game_over': False,
        'last_human': None,
        'last_ai': None,
        'human_is_p1': True,
        'stats': [0, 0, 0],  # wins, losses, draws
    }


def new_game(state, play_as):
    if state is None:
        state = create_initial_state()
    else:
        env = ToguzKumalakEnv(num_stack=NUM_STACK)
        env.reset()
        state['env'] = env
        state['game_over'] = False
        state['last_human'] = None
        state['last_ai'] = None

    state['human_is_p1'] = (play_as == "Player 1 (moves first)")

    env = state['env']
    human_is_p1 = state['human_is_p1']
    human_id = env.black_player if human_is_p1 else env.white_player

    # If AI moves first
    if env.to_play != human_id:
        move = ai_move(env, num_simulations=50, num_parallel=4)
        env.step(move)
        state['last_ai'] = move

    msg = "Your turn — click a pit (1-9)"
    html = render_board(env, msg, state['last_human'], state['last_ai'],
                        False, human_is_p1)
    w, l, d = state['stats']
    stats_text = f"Wins: {w} | Losses: {l} | Draws: {d}"
    return state, html, stats_text


def make_move(pit_index, state, num_sims):
    if state is None:
        state = create_initial_state()
        html = render_board(state['env'], "Click 'New Game' to start!",
                            game_over=False, human_is_p1=True)
        return state, html, "Wins: 0 | Losses: 0 | Draws: 0"

    env = state['env']
    human_is_p1 = state['human_is_p1']
    human_id = env.black_player if human_is_p1 else env.white_player
    ai_id = env.white_player if human_is_p1 else env.black_player

    if state['game_over']:
        html = render_board(env, "Game over! Click 'New Game' to play again.",
                            state['last_human'], state['last_ai'],
                            True, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w} | Losses: {l} | Draws: {d}"

    if env.to_play != human_id:
        html = render_board(env, "Not your turn!",
                            state['last_human'], state['last_ai'],
                            False, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w} | Losses: {l} | Draws: {d}"

    if not env.legal_actions[pit_index]:
        html = render_board(env, f"Pit {pit_index+1} is not a legal move! Choose another.",
                            state['last_human'], state['last_ai'],
                            False, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w} | Losses: {l} | Draws: {d}"

    # Human move
    env.step(pit_index)
    state['last_human'] = pit_index

    if env.is_game_over():
        state['game_over'] = True
        msg = _result_message(env, human_id)
        _update_stats(state, env, human_id)
        html = render_board(env, msg, state['last_human'], state['last_ai'],
                            True, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w} | Losses: {l} | Draws: {d}"

    # AI move
    num_sims_int = int(num_sims)
    move = ai_move(env, num_simulations=num_sims_int, num_parallel=min(8, num_sims_int))
    env.step(move)
    state['last_ai'] = move

    if env.is_game_over():
        state['game_over'] = True
        msg = _result_message(env, human_id)
        _update_stats(state, env, human_id)
        html = render_board(env, msg, state['last_human'], state['last_ai'],
                            True, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w} | Losses: {l} | Draws: {d}"

    msg = "Your turn — click a pit (1-9)"
    html = render_board(env, msg, state['last_human'], state['last_ai'],
                        False, human_is_p1)
    w, l, d = state['stats']
    return state, html, f"Wins: {w} | Losses: {l} | Draws: {d}"


def _result_message(env, human_id):
    if env.winner == human_id:
        return f"You win! {env.kazans[0]}-{env.kazans[1]}"
    elif env.winner == -1:
        return f"Draw! {env.kazans[0]}-{env.kazans[1]}"
    else:
        return f"AI wins! {env.kazans[0]}-{env.kazans[1]}"


def _update_stats(state, env, human_id):
    if env.winner == human_id:
        state['stats'][0] += 1
    elif env.winner == -1:
        state['stats'][2] += 1
    else:
        state['stats'][1] += 1


# ── Gradio UI ──

with gr.Blocks(
    title="Togyz Kumalak — AlphaZero",
    theme=gr.themes.Base(
        primary_hue="amber",
        neutral_hue="stone",
    ),
    css="""
    .pit-btn { min-width: 60px !important; height: 45px !important; font-size: 16px !important; font-weight: bold !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # Togyz Kumalak — Play Against AlphaZero AI
    Traditional Central Asian mancala game. First to **82 stones** wins!
    Pick up stones from one of your 9 pits and sow them counter-clockwise.
    [Rules](https://en.wikipedia.org/wiki/Toguz_korgol) |
    [Source Code](https://github.com/Eraly-ml/alpha_zero_toguz_Qumalaq)
    """)

    game_state = gr.State(None)

    board_html = gr.HTML(
        render_board(env_template, "Click 'New Game' to start!", game_over=True, human_is_p1=True),
    )

    gr.Markdown("### Click a pit to make your move:")

    with gr.Row():
        pit_buttons = []
        for i in range(9):
            btn = gr.Button(f"Pit {i+1}", elem_classes="pit-btn", variant="secondary")
            pit_buttons.append(btn)

    with gr.Row():
        with gr.Column(scale=1):
            play_as = gr.Radio(
                ["Player 1 (moves first)", "Player 2"],
                value="Player 1 (moves first)",
                label="Play as",
            )
        with gr.Column(scale=1):
            difficulty = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="AI Strength (MCTS simulations)",
                info="Higher = stronger but slower. 50 is ~3s per move on CPU.",
            )
        with gr.Column(scale=1):
            new_game_btn = gr.Button("New Game", variant="primary", size="lg")
            stats_display = gr.Textbox(
                value="Wins: 0 | Losses: 0 | Draws: 0",
                label="Session Stats",
                interactive=False,
            )

    # Wire events
    new_game_btn.click(
        fn=new_game,
        inputs=[game_state, play_as],
        outputs=[game_state, board_html, stats_display],
    )

    for i, btn in enumerate(pit_buttons):
        btn.click(
            fn=make_move,
            inputs=[gr.Number(value=i, visible=False), game_state, difficulty],
            outputs=[game_state, board_html, stats_display],
        )

    gr.Markdown("""
    ---
    **About:** AlphaZero neural network (~1.8M parameters) trained entirely through self-play.
    No human game data was used. Built with PyTorch + MCTS.
    """)


if __name__ == "__main__":
    demo.launch()
