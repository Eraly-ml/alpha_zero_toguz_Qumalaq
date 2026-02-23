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


# ══════════════════════════════════════════
#  HTML / CSS Board Renderer
# ══════════════════════════════════════════

BOARD_CSS = """
<style>
/* ── Reset & root ── */
.tq-wrap{
    --pit-w:58px; --pit-h:72px; --stone-sz:9px; --gap:3px;
    --c-bg:#1e1e1e; --c-board:#2c2a26; --c-green:#4a7c59; --c-red:#7c4a4a;
    --c-kazan-g:#3d6b4a; --c-kazan-r:#6b3d3d; --c-gold:#e8a838;
    --c-txt:#e0e0e0; --c-dim:#777; --c-stone1:#f0deb4; --c-stone2:#c4a265;
    --c-glow-you:rgba(74,200,110,.55); --c-glow-ai:rgba(230,90,90,.55);
    font-family:'Segoe UI',system-ui,sans-serif;
    background:var(--c-bg); border-radius:14px; padding:14px 10px;
    max-width:750px; margin:0 auto; box-sizing:border-box;
    user-select:none; overflow-x:auto;
}
/* ── Title & status ── */
.tq-title{text-align:center;color:var(--c-gold);font-size:1.3rem;font-weight:700;margin:0 0 2px}
.tq-status{text-align:center;font-size:.95rem;margin:2px 0 10px;min-height:1.4em;font-weight:600}
/* ── Main board flex ── */
.tq-main{display:flex;align-items:center;justify-content:center;gap:6px}
/* ── Kazan (store) ── */
.tq-kazan{
    min-width:60px;width:60px;border-radius:12px;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    border:2px solid #555;padding:8px 0;
}
.tq-kazan-num{font-size:1.5rem;font-weight:800;color:var(--c-txt)}
.tq-kazan-label{font-size:.6rem;color:var(--c-dim);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
/* ── Pits container ── */
.tq-pits{display:flex;flex-direction:column;gap:4px}
.tq-row-label{text-align:center;font-size:.7rem;color:var(--c-dim);margin:1px 0}
.tq-row{display:flex;gap:var(--gap);justify-content:center}
.tq-divider{border-top:1px dashed #444;margin:1px 0}
/* ── Individual pit ── */
.tq-pit{
    position:relative;width:var(--pit-w);height:var(--pit-h);
    border-radius:10px;display:flex;flex-direction:column;
    align-items:center;justify-content:flex-start;
    padding-top:4px;box-sizing:border-box;
    border:2px solid #444;transition:border .15s,box-shadow .15s;
}
.tq-pit-you{background:var(--c-green)}
.tq-pit-ai{background:var(--c-red)}
/* Move glow */
.tq-pit.tq-last-you{border-color:#4ac86e;box-shadow:0 0 10px var(--c-glow-you)}
.tq-pit.tq-last-ai{border-color:#e65a5a;box-shadow:0 0 10px var(--c-glow-ai)}
/* Tuz marker */
.tq-tuz{
    position:absolute;top:2px;right:3px;
    background:var(--c-gold);color:#1e1e1e;font-size:.55rem;font-weight:800;
    width:16px;height:16px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    box-shadow:0 0 4px var(--c-gold);
}
/* ── Stones grid inside pit ── */
.tq-stones{
    display:flex;flex-wrap:wrap;justify-content:center;align-items:center;
    gap:2px;padding:0 3px;max-width:calc(var(--pit-w) - 8px);
    min-height:28px;
}
.tq-stone{
    width:var(--stone-sz);height:var(--stone-sz);border-radius:50%;
    background:radial-gradient(circle at 35% 30%,var(--c-stone1),var(--c-stone2));
    box-shadow:inset -1px -1px 2px rgba(0,0,0,.35),0 1px 1px rgba(0,0,0,.2);
    flex-shrink:0;
}
.tq-stone.tq-stone-many{
    width:7px;height:7px;
}
/* Count badge */
.tq-count{
    font-size:.8rem;font-weight:800;color:var(--c-txt);
    line-height:1;margin-top:auto;padding-bottom:2px;
}
/* Pit number label */
.tq-pit-num{
    font-size:.55rem;color:rgba(255,255,255,.35);
    position:absolute;bottom:1px;right:4px;
}
/* Move arrow badge */
.tq-move-badge{
    position:absolute;top:-14px;left:50%;transform:translateX(-50%);
    font-size:.55rem;font-weight:700;white-space:nowrap;
    padding:1px 5px;border-radius:4px;
}
.tq-move-badge.you{background:#4ac86e;color:#111}
.tq-move-badge.ai{background:#e65a5a;color:#fff}
/* ── Score bar ── */
.tq-score{
    text-align:center;margin-top:8px;font-size:.75rem;color:var(--c-dim);
}
.tq-progress{
    margin:4px auto 0;width:80%;max-width:350px;height:8px;
    background:#333;border-radius:4px;overflow:hidden;display:flex;
}
.tq-prog-you{background:var(--c-green);transition:width .3s}
.tq-prog-ai{background:var(--c-red);transition:width .3s}

/* ── Responsive ── */
@media(max-width:640px){
    .tq-wrap{--pit-w:9.5vw;--pit-h:12vw;--stone-sz:clamp(5px,1.3vw,8px);--gap:1.5px;padding:10px 4px}
    .tq-kazan{min-width:11vw;width:11vw;padding:6px 0}
    .tq-kazan-num{font-size:1.1rem}
    .tq-title{font-size:1rem}
    .tq-status{font-size:.8rem}
    .tq-count{font-size:.65rem}
    .tq-tuz{width:13px;height:13px;font-size:.5rem}
    .tq-move-badge{font-size:.45rem;top:-11px}
}
@media(max-width:400px){
    .tq-wrap{--pit-w:9vw;--pit-h:11.5vw;--stone-sz:clamp(4px,1.1vw,7px);padding:8px 2px}
    .tq-kazan{min-width:10vw;width:10vw}
    .tq-kazan-num{font-size:.95rem}
    .tq-pit-num{display:none}
    .tq-row-label{font-size:.6rem}
}
</style>
"""


def _render_stones_html(count):
    """Render stone circles inside a pit."""
    if count == 0:
        return '<div class="tq-stones"></div><span class="tq-count">0</span>'

    visible = min(count, 12)
    cls = "tq-stone" if count <= 9 else "tq-stone tq-stone-many"
    dots = ''.join(f'<span class="{cls}"></span>' for _ in range(visible))
    if count > 12:
        dots += '<span class="tq-stone tq-stone-many" style="opacity:.4"></span>'

    return f'<div class="tq-stones">{dots}</div><span class="tq-count">{count}</span>'


def render_board(env, message="", last_human=None, last_ai=None,
                 game_over=False, human_is_p1=True):
    board = env.board
    kazans = env.kazans
    tuzduks = env.tuzduks

    if human_is_p1:
        top_p, bot_p = 1, 0
        top_label, bot_label = "AI", "YOU"
        top_kazan, bot_kazan = kazans[1], kazans[0]
        top_tuz_owner, bot_tuz_owner = 0, 1
    else:
        top_p, bot_p = 0, 1
        top_label, bot_label = "AI", "YOU"
        top_kazan, bot_kazan = kazans[0], kazans[1]
        top_tuz_owner, bot_tuz_owner = 1, 0

    def pit(player_row, pit_i, is_top, last_move_you, last_move_ai, tuz_owner):
        stones = int(board[player_row, pit_i])
        is_tuz = tuzduks[tuz_owner] == pit_i

        side_cls = "tq-pit-ai" if is_top else "tq-pit-you"
        glow_cls = ""
        badge = ""
        if last_move_you is not None and not is_top and last_move_you == pit_i:
            glow_cls = " tq-last-you"
            badge = '<span class="tq-move-badge you">YOUR MOVE</span>'
        elif last_move_ai is not None and is_top and last_move_ai == pit_i:
            glow_cls = " tq-last-ai"
            badge = '<span class="tq-move-badge ai">AI MOVE</span>'

        tuz_html = '<span class="tq-tuz">TUZ</span>' if is_tuz else ''

        inner = _render_stones_html(stones)
        num_label = f'<span class="tq-pit-num">{pit_i+1}</span>'

        return (
            f'<div class="tq-pit {side_cls}{glow_cls}">'
            f'{badge}{tuz_html}{inner}{num_label}'
            f'</div>'
        )

    # Top row: reversed (pit 9 on left → pit 1 on right)
    top_pits = ''.join(
        pit(top_p, 8 - i, True, last_human, last_ai, top_tuz_owner)
        for i in range(9)
    )
    # Bottom row: normal (pit 1 on left → pit 9 on right)
    bot_pits = ''.join(
        pit(bot_p, i, False, last_human, last_ai, bot_tuz_owner)
        for i in range(9)
    )

    # Status color
    msg_color = "var(--c-gold)"
    if "ai wins" in message.lower():
        msg_color = "#e65a5a"
    elif "you win" in message.lower():
        msg_color = "#4ac86e"
    elif "draw" in message.lower():
        msg_color = "var(--c-dim)"

    # Progress bar
    total = int(kazans[0]) + int(kazans[1])
    you_k = int(bot_kazan)
    ai_k = int(top_kazan)
    you_pct = (you_k / 162 * 100) if total > 0 else 0
    ai_pct = (ai_k / 162 * 100) if total > 0 else 0

    # Direction arrows
    top_dir = "&#8592; sowing direction &#8592;" if top_p == 1 else "&#8594; sowing direction &#8594;"
    bot_dir = "&#8594; sowing direction &#8594;" if bot_p == 0 else "&#8592; sowing direction &#8592;"

    html = f"""{BOARD_CSS}
<div class="tq-wrap">
  <div class="tq-title">&#127922; Togyz Kumalak</div>
  <div class="tq-status" style="color:{msg_color}">{message}</div>

  <div class="tq-main">
    <!-- AI Kazan -->
    <div class="tq-kazan" style="background:{'var(--c-kazan-r)' if top_p==1 else 'var(--c-kazan-g)'}">
      <span class="tq-kazan-num">{top_kazan}</span>
      <span class="tq-kazan-label">{top_label}</span>
    </div>

    <div class="tq-pits">
      <div class="tq-row-label">{top_label} pits &nbsp; {top_dir}</div>
      <div class="tq-row">{top_pits}</div>
      <div class="tq-divider"></div>
      <div class="tq-row">{bot_pits}</div>
      <div class="tq-row-label">{bot_label} pits &nbsp; {bot_dir}</div>
    </div>

    <!-- Your Kazan -->
    <div class="tq-kazan" style="background:{'var(--c-kazan-g)' if bot_p==0 else 'var(--c-kazan-r)'}">
      <span class="tq-kazan-num">{bot_kazan}</span>
      <span class="tq-kazan-label">{bot_label}</span>
    </div>
  </div>

  <div class="tq-score">
    You <b>{you_k}</b> / 82 &nbsp;&mdash;&nbsp; AI <b>{ai_k}</b> / 82
    <div class="tq-progress">
      <div class="tq-prog-you" style="width:{you_pct:.1f}%"></div>
      <div style="flex:1"></div>
      <div class="tq-prog-ai" style="width:{ai_pct:.1f}%"></div>
    </div>
  </div>
</div>"""
    return html


# ══════════════════════════════════════════
#  Game State
# ══════════════════════════════════════════

def create_initial_state():
    env = ToguzKumalakEnv(num_stack=NUM_STACK)
    env.reset()
    return {
        'env': env,
        'game_over': False,
        'last_human': None,
        'last_ai': None,
        'human_is_p1': True,
        'stats': [0, 0, 0],
    }


def new_game(state, play_as, num_sims):
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

    if env.to_play != human_id:
        n = max(10, int(num_sims))
        move = ai_move(env, num_simulations=n, num_parallel=min(8, n))
        env.step(move)
        state['last_ai'] = move

    msg = "Your turn — pick a pit below (1-9)"
    html = render_board(env, msg, state['last_human'], state['last_ai'],
                        False, human_is_p1)
    w, l, d = state['stats']
    return state, html, f"Wins: {w}  |  Losses: {l}  |  Draws: {d}"


def make_move(pit_index, state, num_sims):
    if state is None:
        state = create_initial_state()
        html = render_board(state['env'], "Click 'New Game' to start!",
                            game_over=False, human_is_p1=True)
        return state, html, "Wins: 0  |  Losses: 0  |  Draws: 0"

    env = state['env']
    human_is_p1 = state['human_is_p1']
    human_id = env.black_player if human_is_p1 else env.white_player

    if state['game_over']:
        html = render_board(env, "Game over! Click 'New Game' to play again.",
                            state['last_human'], state['last_ai'], True, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w}  |  Losses: {l}  |  Draws: {d}"

    if env.to_play != human_id:
        html = render_board(env, "Wait — it's not your turn.",
                            state['last_human'], state['last_ai'], False, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w}  |  Losses: {l}  |  Draws: {d}"

    if not env.legal_actions[pit_index]:
        html = render_board(env, f"Pit {pit_index+1} is empty or blocked! Pick another.",
                            state['last_human'], state['last_ai'], False, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w}  |  Losses: {l}  |  Draws: {d}"

    # Human move
    env.step(pit_index)
    state['last_human'] = pit_index

    if env.is_game_over():
        state['game_over'] = True
        msg = _result_message(env, human_id)
        _update_stats(state, env, human_id)
        html = render_board(env, msg, state['last_human'], state['last_ai'], True, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w}  |  Losses: {l}  |  Draws: {d}"

    # AI move
    n = max(10, int(num_sims))
    move = ai_move(env, num_simulations=n, num_parallel=min(8, n))
    env.step(move)
    state['last_ai'] = move

    if env.is_game_over():
        state['game_over'] = True
        msg = _result_message(env, human_id)
        _update_stats(state, env, human_id)
        html = render_board(env, msg, state['last_human'], state['last_ai'], True, human_is_p1)
        w, l, d = state['stats']
        return state, html, f"Wins: {w}  |  Losses: {l}  |  Draws: {d}"

    msg = "Your turn — pick a pit below (1-9)"
    html = render_board(env, msg, state['last_human'], state['last_ai'], False, human_is_p1)
    w, l, d = state['stats']
    return state, html, f"Wins: {w}  |  Losses: {l}  |  Draws: {d}"


def _result_message(env, human_id):
    s = f"{env.kazans[0]} - {env.kazans[1]}"
    if env.winner == human_id:
        return f"You win! ({s})"
    elif env.winner == -1:
        return f"Draw! ({s})"
    else:
        return f"AI wins! ({s})"


def _update_stats(state, env, human_id):
    if env.winner == human_id:
        state['stats'][0] += 1
    elif env.winner == -1:
        state['stats'][2] += 1
    else:
        state['stats'][1] += 1


# ══════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════

RULES_MD = """
## How to Play Togyz Kumalak

**Togyz Kumalak** (Тоғыз құмалақ) is a traditional board game from Central Asia (Kazakhstan, Kyrgyzstan).
Think of it as **strategic Mancala** — but with deeper tactics!

---

### Setup
| | |
|---|---|
| **Board** | 2 rows of **9 pits** (called *otau*) + 2 scoring pits (*kazan*) |
| **Stones** | **162 total** — 9 stones in each pit at the start |
| **Goal** | Be the first to collect **82 or more** stones in your kazan |

---

### Your Turn — Sowing
1. **Pick** one of your 9 pits that has stones
2. All stones from that pit are distributed **one by one**, counter-clockwise:
   - Your pits go **left → right** (pits 1-9)
   - Then opponent's pits go **left → right** (pits 1-9), and back to yours

3. **Special rule:** If the pit has **more than 1 stone**, leave 1 stone behind in the original pit and sow the rest

4. If the pit has **exactly 1 stone**, simply move it to the next pit (nothing stays behind)

---

### Captures
After sowing, look at where your **last stone** landed:

| Condition | What happens |
|-----------|-------------|
| Last stone lands in **opponent's pit** making it **even** (2, 4, 6, ...) | You **capture all** stones from that pit → your kazan |
| Last stone lands in **opponent's pit** making it exactly **3** | You may declare it a **Tuzdyk** (see below) |

---

### Tuzdyk (Tuz) — Permanent Capture
If your last stone makes an **opponent's pit = 3**, you can turn it into your **Tuzdyk**:
- Marked with a golden **TUZ** badge on the board
- From now on, **all stones** that land in this pit go directly to **your kazan**
- The opponent can never pick up from their tuzdyk pit

**Rules for Tuzdyk:**
- Each player can have **at most 1** tuzdyk
- **Pit 9** (the last pit) **cannot** become a tuzdyk
- Both players **cannot** tuzdyk the **same-numbered** pit

---

### Winning
- First player to reach **82+ stones** wins immediately
- If both reach **81**, it's a **draw**
- If all pits are empty (no legal moves), remaining stones go to their respective kazans and highest score wins

---

### Quick Tips
- **Watch the progress bar** at the bottom — it shows who's closer to 82
- **Green glow** = your last move, **Red glow** = AI's last move
- **Tuzdyk is powerful** — creating one early can give you a huge advantage
- Think about **which pits you leave vulnerable** to capture (even counts!)
- Pits with many stones can **"wrap around"** the entire board
"""

with gr.Blocks(
    title="Togyz Kumalak — AlphaZero",
    theme=gr.themes.Base(primary_hue="amber", neutral_hue="stone"),
    css="""
    .pit-btn {
        min-width:0!important; padding:6px 0!important;
        font-size:15px!important; font-weight:700!important;
        flex:1 1 0!important;
    }
    footer{display:none!important}
    .contain{max-width:800px!important;margin:0 auto!important}
    """
) as demo:

    # ── Header ──
    gr.Markdown("# Togyz Kumalak — Play Against AlphaZero AI")
    gr.Markdown(
        "A traditional Central Asian strategy game. "
        "First to **82 stones** wins! "
        "[Source Code](https://github.com/Eraly-ml/alpha_zero_toguz_Qumalaq)"
    )

    # ── Rules accordion ──
    with gr.Accordion("How to Play (rules & tips)", open=False):
        gr.Markdown(RULES_MD)

    # ── State ──
    game_state = gr.State(None)

    # ── Board ──
    board_html = gr.HTML(
        render_board(env_template, "Click 'New Game' to start!", game_over=True, human_is_p1=True),
    )

    # ── Pit buttons ──
    gr.Markdown("**Pick a pit to move** (your row, left to right):")
    with gr.Row(equal_height=True):
        pit_buttons = []
        for i in range(9):
            btn = gr.Button(f"{i+1}", elem_classes="pit-btn", variant="secondary", size="sm")
            pit_buttons.append(btn)

    # ── Controls ──
    with gr.Row():
        with gr.Column(scale=1, min_width=140):
            new_game_btn = gr.Button("New Game", variant="primary", size="lg")
            stats_display = gr.Textbox(
                value="Wins: 0  |  Losses: 0  |  Draws: 0",
                label="Session Stats", interactive=False, max_lines=1,
            )
        with gr.Column(scale=1, min_width=140):
            play_as = gr.Radio(
                ["Player 1 (moves first)", "Player 2"],
                value="Player 1 (moves first)",
                label="Play as",
            )
        with gr.Column(scale=1, min_width=140):
            difficulty = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="AI Strength (MCTS simulations)",
                info="Higher = stronger but slower",
            )

    # ── Wiring ──
    new_game_btn.click(
        fn=new_game,
        inputs=[game_state, play_as, difficulty],
        outputs=[game_state, board_html, stats_display],
    )

    for i, btn in enumerate(pit_buttons):
        btn.click(
            fn=make_move,
            inputs=[gr.Number(value=i, visible=False), game_state, difficulty],
            outputs=[game_state, board_html, stats_display],
        )

    # ── Footer ──
    gr.Markdown(
        "---\n"
        "**About the AI:** AlphaZero neural network (~1.8M parameters) trained entirely "
        "through self-play — zero human game data. Uses Monte Carlo Tree Search + deep "
        "residual network. Built with PyTorch."
    )


if __name__ == "__main__":
    demo.launch()
