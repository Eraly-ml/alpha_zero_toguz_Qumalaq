#!/usr/bin/env python3
"""Togyz Kumalak GUI — play against the AlphaZero agent.

Usage:
    python play_gui.py --ckpt checkpoints/training_steps_4800.ckpt
    python play_gui.py --ckpt checkpoints/training_steps_4800.ckpt --human_color white
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_stack', 4, 'Stack N previous states.')
flags.DEFINE_integer('num_res_blocks', 6, 'Number of residual blocks.')
flags.DEFINE_integer('num_filters', 128, 'Number of conv2d filters.')
flags.DEFINE_integer('num_fc_units', 256, 'Number of hidden units in FC layer.')
flags.DEFINE_bool('use_se', False, 'Use SE attention in residual blocks.')
flags.DEFINE_string('ckpt', '', 'Path to checkpoint file.')
flags.DEFINE_integer('num_simulations', 200, 'MCTS simulations per move.')
flags.DEFINE_integer('num_parallel', 8, 'Parallel MCTS leaves.')
flags.DEFINE_float('c_puct_base', 19652, 'PUCT exploration base.')
flags.DEFINE_float('c_puct_init', 1.25, 'PUCT exploration init.')
flags.DEFINE_string('human_color', 'black', 'Human plays as "black" (P1, first) or "white" (P2).')
flags.DEFINE_integer('seed', 1, 'Random seed.')

FLAGS(sys.argv)

import torch
import numpy as np
from alpha_zero.envs.toguz import ToguzKumalakEnv
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.pipeline import create_mcts_player, set_seed, disable_auto_grad

# ─── Colors ───
BG = '#2b2b2b'
BOARD_BG = '#3c3a36'
PIT_COLOR_P1 = '#4a7c59'
PIT_COLOR_P2 = '#7c4a4a'
PIT_HOVER = '#5a9a6e'
PIT_DISABLED = '#555555'
KAZAN_P1 = '#3d6b4a'
KAZAN_P2 = '#6b3d3d'
TUZ_MARKER = '#ffcc00'
TEXT_COLOR = '#e0e0e0'
TEXT_DIM = '#888888'
STONE_COLOR = '#ddd5c0'
ACCENT = '#e8a838'
LAST_MOVE = '#e8a838'


class ToguzGUI:
    PIT_W = 70
    PIT_H = 90
    KAZAN_W = 90
    GAP = 6
    MARGIN = 20

    def __init__(self):
        set_seed(FLAGS.seed)

        # ─── Load AI ───
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = ToguzKumalakEnv(num_stack=FLAGS.num_stack)
        input_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n

        net = AlphaZeroNet(input_shape, num_actions,
                           FLAGS.num_res_blocks, FLAGS.num_filters,
                           FLAGS.num_fc_units, False, use_se=FLAGS.use_se)
        if FLAGS.ckpt and os.path.isfile(FLAGS.ckpt):
            state = torch.load(FLAGS.ckpt, map_location=self.device)
            net.load_state_dict(state['network'])
            print(f'Loaded checkpoint: {FLAGS.ckpt}')
        else:
            print('WARNING: No checkpoint loaded, AI uses random weights!')
        net.to(self.device)
        net.eval()
        disable_auto_grad(net)

        self.ai_player = create_mcts_player(
            network=net, device=self.device,
            num_simulations=FLAGS.num_simulations,
            num_parallel=FLAGS.num_parallel,
            root_noise=False, deterministic=False,
        )

        self.human_is_black = FLAGS.human_color.lower() == 'black'
        self.human_id = self.env.black_player if self.human_is_black else self.env.white_player
        self.ai_id = self.env.white_player if self.human_is_black else self.env.black_player

        # Stats
        self.human_wins = 0
        self.ai_wins = 0
        self.draws = 0
        self.game_count = 0

        # State
        self.ai_thinking = False
        self.game_over = False
        self.last_ai_move = None
        self.last_human_move = None

        self._build_window()
        self._new_game()

    # ═══════════════════════════════════════════
    #  GUI LAYOUT
    # ═══════════════════════════════════════════
    def _build_window(self):
        self.root = tk.Tk()
        self.root.title('Тоғыз Құмалақ — AlphaZero')
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # Fonts
        self.font_title = tkfont.Font(family='Helvetica', size=16, weight='bold')
        self.font_big = tkfont.Font(family='Helvetica', size=14, weight='bold')
        self.font_num = tkfont.Font(family='Helvetica', size=13, weight='bold')
        self.font_label = tkfont.Font(family='Helvetica', size=10)
        self.font_small = tkfont.Font(family='Helvetica', size=9)
        self.font_status = tkfont.Font(family='Helvetica', size=12)

        # Title
        tk.Label(self.root, text='Тоғыз Құмалақ', font=self.font_title,
                 bg=BG, fg=ACCENT).pack(pady=(10, 2))

        # Status
        self.status_var = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self.status_var, font=self.font_status,
                 bg=BG, fg=TEXT_COLOR).pack(pady=(0, 5))

        # Canvas
        cw = self.MARGIN * 2 + self.KAZAN_W + self.GAP + 9 * (self.PIT_W + self.GAP) + self.KAZAN_W
        ch = self.MARGIN * 2 + 2 * self.PIT_H + 30 + 20
        self.canvas = tk.Canvas(self.root, width=cw, height=ch, bg=BOARD_BG,
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=5)
        self.canvas.bind('<Motion>', self._on_hover)
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas_w = cw
        self.canvas_h = ch

        # Score bar
        score_frame = tk.Frame(self.root, bg=BG)
        score_frame.pack(pady=5)
        self.score_var = tk.StringVar(value='')
        tk.Label(score_frame, textvariable=self.score_var, font=self.font_label,
                 bg=BG, fg=TEXT_DIM).pack()

        # Buttons
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=(0, 10))
        tk.Button(btn_frame, text='Жаңа ойын', font=self.font_label,
                  command=self._new_game, bg='#444', fg=TEXT_COLOR,
                  activebackground='#666', activeforeground=TEXT_COLOR,
                  relief='flat', padx=15, pady=5).pack(side='left', padx=5)
        tk.Button(btn_frame, text='Шығу', font=self.font_label,
                  command=self.root.destroy, bg='#544', fg=TEXT_COLOR,
                  activebackground='#766', activeforeground=TEXT_COLOR,
                  relief='flat', padx=15, pady=5).pack(side='left', padx=5)

        # Pit hitboxes: list of (x1, y1, x2, y2, pit_index, player_row)
        self._pit_rects = []
        self._hover_pit = None

    # ═══════════════════════════════════════════
    #  DRAWING
    # ═══════════════════════════════════════════
    def _draw_board(self):
        c = self.canvas
        c.delete('all')
        self._pit_rects.clear()

        board = self.env.board
        kazans = self.env.kazans
        tuzduks = self.env.tuzduks
        current_player = self.env.to_play

        # Determine which row is top/bottom based on human color
        # Human is always at the bottom
        if self.human_is_black:
            top_player = 1   # P2 (AI) on top
            bot_player = 0   # P1 (Human) on bottom
            top_label = 'AI (Ойыншы 2)'
            bot_label = 'Сіз (Ойыншы 1)'
        else:
            top_player = 0   # P1 (AI) on top
            bot_player = 1   # P2 (Human) on bottom
            top_label = 'AI (Ойыншы 1)'
            bot_label = 'Сіз (Ойыншы 2)'

        M = self.MARGIN
        KW = self.KAZAN_W
        PW = self.PIT_W
        PH = self.PIT_H
        G = self.GAP

        # ── Kazans ──
        # Left kazan = top player's kazan
        kx1 = M
        ky1 = M
        kx2 = M + KW
        ky2 = M + 2 * PH + 30
        c.create_rectangle(kx1, ky1, kx2, ky2, fill=KAZAN_P2 if top_player == 1 else KAZAN_P1,
                           outline='#555', width=2)
        c.create_text((kx1 + kx2) / 2, (ky1 + ky2) / 2 - 10,
                       text=str(kazans[top_player]), font=self.font_big, fill=TEXT_COLOR)
        c.create_text((kx1 + kx2) / 2, (ky1 + ky2) / 2 + 15,
                       text='Қазан', font=self.font_small, fill=TEXT_DIM)

        # Right kazan = bottom player's kazan
        rx1 = M + KW + G + 9 * (PW + G)
        ry1 = M
        rx2 = rx1 + KW
        ry2 = ry1 + 2 * PH + 30
        c.create_rectangle(rx1, ry1, rx2, ry2, fill=KAZAN_P1 if bot_player == 0 else KAZAN_P2,
                           outline='#555', width=2)
        c.create_text((rx1 + rx2) / 2, (ry1 + ry2) / 2 - 10,
                       text=str(kazans[bot_player]), font=self.font_big, fill=TEXT_COLOR)
        c.create_text((rx1 + rx2) / 2, (ry1 + ry2) / 2 + 15,
                       text='Қазан', font=self.font_small, fill=TEXT_DIM)

        pits_x0 = M + KW + G

        # ── Top row (opponent, shown reversed: pit 8 on left, pit 0 on right) ──
        ty = M
        c.create_text(pits_x0 + 4.5 * (PW + G) - G / 2, ty - 5,
                       text=top_label, font=self.font_small, fill=TEXT_DIM, anchor='s')
        for vis_i in range(9):
            pit_i = 8 - vis_i  # reverse display
            x1 = pits_x0 + vis_i * (PW + G)
            y1 = ty
            x2 = x1 + PW
            y2 = y1 + PH

            is_tuz = (tuzduks[bot_player] == pit_i)
            is_last_move = (self.last_ai_move == pit_i and not self.human_is_black) or \
                           (self.last_ai_move == pit_i and self.human_is_black and top_player == 1)
            # For AI: last_ai_move refers to the AI's pit
            if top_player == 1:
                is_last = self.last_ai_move == pit_i if self.human_is_black else self.last_human_move == pit_i
            else:
                is_last = self.last_ai_move == pit_i if not self.human_is_black else self.last_human_move == pit_i

            fill = PIT_COLOR_P2 if top_player == 1 else PIT_COLOR_P1
            outline = LAST_MOVE if is_last else '#555'
            ow = 3 if is_last else 1

            c.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=ow)
            c.create_text((x1 + x2) / 2, (y1 + y2) / 2 - 5,
                           text=str(board[top_player, pit_i]), font=self.font_num, fill=TEXT_COLOR)
            c.create_text((x1 + x2) / 2, y2 - 12,
                           text=str(pit_i + 1), font=self.font_small, fill=TEXT_DIM)
            if is_tuz:
                c.create_text(x2 - 8, y1 + 8, text='T', font=self.font_small, fill=TUZ_MARKER)

        # ── Separator ──
        sep_y = M + PH + 5
        c.create_line(pits_x0, sep_y + 10, pits_x0 + 9 * (PW + G) - G, sep_y + 10,
                       fill='#555', width=1, dash=(4, 4))

        # ── Bottom row (human's pits, left to right: pit 0 to pit 8) ──
        by = M + PH + 20
        c.create_text(pits_x0 + 4.5 * (PW + G) - G / 2, by + PH + 15,
                       text=bot_label, font=self.font_small, fill=TEXT_DIM, anchor='n')

        is_human_turn = (current_player == self.human_id) and not self.game_over and not self.ai_thinking

        for pit_i in range(9):
            x1 = pits_x0 + pit_i * (PW + G)
            y1 = by
            x2 = x1 + PW
            y2 = y1 + PH

            is_tuz = (tuzduks[top_player] == pit_i)
            is_legal = bool(self.env.legal_actions[pit_i]) and is_human_turn and \
                       current_player == self.human_id
            is_hover = (self._hover_pit == pit_i) and is_legal

            if bot_player == 0:
                is_last = self.last_human_move == pit_i if self.human_is_black else self.last_ai_move == pit_i
            else:
                is_last = self.last_human_move == pit_i if not self.human_is_black else self.last_ai_move == pit_i

            if is_hover:
                fill = PIT_HOVER
            elif is_legal:
                fill = PIT_COLOR_P1 if bot_player == 0 else PIT_COLOR_P2
            else:
                fill = PIT_DISABLED if (is_human_turn and not is_legal and board[bot_player, pit_i] == 0) else \
                       (PIT_COLOR_P1 if bot_player == 0 else PIT_COLOR_P2)

            outline = LAST_MOVE if is_last else '#555'
            ow = 3 if is_last else 1

            c.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=ow)
            c.create_text((x1 + x2) / 2, (y1 + y2) / 2 - 5,
                           text=str(board[bot_player, pit_i]), font=self.font_num, fill=TEXT_COLOR)
            c.create_text((x1 + x2) / 2, y2 - 12,
                           text=str(pit_i + 1), font=self.font_small, fill=TEXT_DIM)
            if is_tuz:
                c.create_text(x2 - 8, y1 + 8, text='T', font=self.font_small, fill=TUZ_MARKER)

            if is_human_turn:
                self._pit_rects.append((x1, y1, x2, y2, pit_i))

    def _update_status(self):
        if self.game_over:
            if self.env.winner == self.human_id:
                self.status_var.set(f'Сіз жеңдіңіз!  {self.env.kazans[0]}-{self.env.kazans[1]}')
            elif self.env.winner == -1:
                self.status_var.set(f'Тең!  {self.env.kazans[0]}-{self.env.kazans[1]}')
            else:
                self.status_var.set(f'AI жеңді!  {self.env.kazans[0]}-{self.env.kazans[1]}')
        elif self.ai_thinking:
            self.status_var.set('AI ойлануда...')
        elif self.env.to_play == self.human_id:
            self.status_var.set('Сіздің кезегіңіз — ұяшық таңдаңыз')
        else:
            self.status_var.set('AI кезегі...')

        self.score_var.set(
            f'Ойын #{self.game_count}  |  Сіз: {self.human_wins}   AI: {self.ai_wins}   Тең: {self.draws}'
        )

    # ═══════════════════════════════════════════
    #  GAME LOGIC
    # ═══════════════════════════════════════════
    def _new_game(self):
        self.env.reset()
        self.game_over = False
        self.ai_thinking = False
        self.last_ai_move = None
        self.last_human_move = None
        self.game_count += 1
        self._draw_board()
        self._update_status()

        # If AI moves first
        if self.env.to_play == self.ai_id:
            self._ai_turn()

    def _human_move(self, pit):
        if self.game_over or self.ai_thinking:
            return
        if self.env.to_play != self.human_id:
            return
        if not self.env.legal_actions[pit]:
            return

        self.last_human_move = pit
        _, _, done, _ = self.env.step(pit)
        self.game_over = done

        if done:
            self._record_result()

        self._draw_board()
        self._update_status()

        if not done and self.env.to_play == self.ai_id:
            self.root.after(100, self._ai_turn)

    def _ai_turn(self):
        self.ai_thinking = True
        self._draw_board()
        self._update_status()

        def think():
            move, *_ = self.ai_player(self.env, None, FLAGS.c_puct_base, FLAGS.c_puct_init)
            self.root.after(0, lambda: self._ai_finish(move))

        threading.Thread(target=think, daemon=True).start()

    def _ai_finish(self, move):
        self.ai_thinking = False
        self.last_ai_move = move
        _, _, done, _ = self.env.step(move)
        self.game_over = done

        if done:
            self._record_result()

        self._draw_board()
        self._update_status()

        # If it's still AI's turn (shouldn't happen but just in case)
        if not done and self.env.to_play == self.ai_id:
            self.root.after(100, self._ai_turn)

    def _record_result(self):
        if self.env.winner == self.human_id:
            self.human_wins += 1
        elif self.env.winner == -1:
            self.draws += 1
        else:
            self.ai_wins += 1

    # ═══════════════════════════════════════════
    #  EVENTS
    # ═══════════════════════════════════════════
    def _on_hover(self, event):
        old = self._hover_pit
        self._hover_pit = None
        for x1, y1, x2, y2, pit_i in self._pit_rects:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                if self.env.legal_actions[pit_i]:
                    self._hover_pit = pit_i
                break
        if old != self._hover_pit:
            self._draw_board()

    def _on_click(self, event):
        if self.ai_thinking or self.game_over:
            return
        for x1, y1, x2, y2, pit_i in self._pit_rects:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._human_move(pit_i)
                return

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    gui = ToguzGUI()
    gui.run()
