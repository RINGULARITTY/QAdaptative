import dearpygui.dearpygui as dpg
from typing import List, Tuple, Union
import time
from math import sqrt
import numpy as np

class Entity:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

class Ground:   
    def __init__(self, id, color):
       self.id = id
       self.color = color

class Grounds:
    NONE = Ground(0, (0, 0, 0))
    PATH = Ground(1, (255, 255, 255))
    START = Ground(2, (0, 255, 0))
    END = Ground(3, (255, 0, 0))
    TOWER = Ground(4, (225, 255, 0))

class Entity:
    def __init__(self, id, render_methods: List[exec]):
        self.id = id
        self.render_methods: List[exec] = render_methods
        
        self.x = 0
        self.y = 0
    
    def render(self, ground_size, **params):
        for render_method in self.render_methods:
            render_method(self.y, self.x, ground_size, **params)

class Enemy(Entity):
    def __init__(self, id, level: int, delta: int, render_methods: List[exec]):
        super().__init__(id, render_methods)
        self.max_hp = 5 * level
        self.golds = 2 * level + 5
        self.current_hp = self.max_hp
        self.delta = delta

class Tower(Entity):
    def __init__(self, id, level, x, y, render_methods: List[exec]):
        super().__init__(id, render_methods)
        self.level = level
        self.x = x
        self.y = y
        self.damage = 0
        self.reload_speed_coef = 0
        self.current_reload = 0
        self.range = 0
        self.upgrade_price = 0

        if level != 0:
            self._upgrade_stats()
    
    def _upgrade_stats(self):
        self.damage = 3 * self.level
        self.reload_speed_coef = 2 / (self.level / 2)
        self.range = self.level / 4 + 1.5
        self.upgrade_price = 35 + 5 * self.level
    
    def upgrade(self):
        self.level += 1
        self._upgrade_stats()
    
    def update(self):
        self.current_reload -= 1

    def target(self, enemy: Enemy):
        if self.current_reload <= 0:
            distance = sqrt(pow(enemy.x - self.x, 2) + pow(enemy.y - self.y, 2))
            if distance > self.range:
                return False
            self.current_reload = distance * self.reload_speed_coef
            enemy.current_hp -= self.damage
            return True
        return False

class Towers:   
    BASIC_TOWER = lambda level, x, y: Tower(0, level, x, y,
        [
            lambda x, y, ground_size, **params: dpg.draw_triangle(
                (x * ground_size + ground_size // 2, y * ground_size + ground_size // 4),
                (x * ground_size + ground_size // 4, y * ground_size + 3 * ground_size // 4),
                (x * ground_size + 3 * ground_size // 4, y * ground_size + 3 * ground_size // 4),
                color=(0, 0, 200),
                fill=(0, 0, 200),
                parent="drawing"
            ),
            lambda x, y, ground_size, **params: dpg.draw_circle(
                (x * ground_size + ground_size // 2, y * ground_size + ground_size // 2),
                params["range"] * ground_size,
                color=(100, 100, 100),
                parent="drawing"
            ),
        ]
    )
    ARCHER_TOWER = lambda level, x, y: Tower(1, level, x, y,
        [
            lambda x, y, ground_size, **params: dpg.draw_triangle(
                (x * ground_size + ground_size // 2, y * ground_size + ground_size // 4),
                (x * ground_size + ground_size // 4, y * ground_size + 3 * ground_size // 4),
                (x * ground_size + 3 * ground_size // 4, y * ground_size + 3 * ground_size // 4),
                color=(25, 200, 25),
                fill=(25, 200, 25),
                parent="drawing"
            ),
            lambda x, y, ground_size, **params: dpg.draw_circle(
                (x * ground_size + ground_size // 2, y * ground_size + ground_size // 2),
                params["range"] * ground_size,
                color=(100, 100, 100),
                parent="drawing"
            ),
        ]
    )

class TowersGlossary:
    TOWERS = {
        0: Towers.BASIC_TOWER,
        1: Towers.ARCHER_TOWER
    }

class Enemies:
    BASIC_ENEMY = lambda level: Enemy(0, level, 4,
        [
            lambda x, y, ground_size: dpg.draw_circle(
                (x * ground_size + game.ground_size / 2, y * ground_size + game.ground_size / 2),
                game.ground_size // 5, 
                color=(255, 150, 0),
                fill=(255, 150, 0),
                parent="drawing"
            )
        ]
    )

class Wave:
    def __init__(self, enemies_timer):
        self.timer = 0
        self.step = 0
        #Enemy, spawn frame, current waypoint, current delta
        self.enemies: List[Tuple[Enemy, int, int, int]] = [[*enemy_timer, 0, 0] for enemy_timer in enemies_timer]
        self.wave_id = 0
    
    @staticmethod
    def calculate_progression(enemy: Tuple[Enemy, int, int, int], waypoints, waypoints_progression):
        if enemy is None:
            return -1

        if enemy[2] == len(waypoints):
            return 1

        before, next = waypoints[enemy[2] - 1], waypoints[enemy[2]]
        local_progression = (0 if next[0] - before[0] == 0 else (enemy[0].x - before[0]) / (next[0] - before[0])) + (0 if next[1] - before[1] == 0 else (enemy[0].y - before[1]) / (next[1] - before[1]))
        return waypoints_progression[enemy[2] - 1] + (waypoints_progression[enemy[2]] - waypoints_progression[enemy[2] - 1]) * local_progression

    def get_sorted_enemies(self, waypoints, waypoints_progression):
        return sorted(self.enemies, key=lambda enemy: Wave.calculate_progression(enemy, waypoints, waypoints_progression))
        
    def start(self):
        self.timer = time.time()
        self.step = 0
    
    def update(self, waypoints: List[Tuple[int, int]]) -> int:
        self.step += 1
        base_elements = 0
        killed_enemies = 0
        golds_earned = 0
        for i in range(len(self.enemies)):
            enemy = self.enemies[i - base_elements - killed_enemies]
            if enemy is None:
                continue
            if enemy[0].current_hp <= 0:
                golds_earned += self.enemies[i][0].golds
                self.enemies[i] = None
                killed_enemies += 1
            elif enemy[2] == 0 and self.step >= enemy[1]:
                enemy[2] = 1
                enemy[0].x = waypoints[0][0]; enemy[0].y = waypoints[0][1]
            elif enemy[2] > 0 and enemy[2] < len(waypoints):
                enemy[3] += 1

                if enemy[3] >= enemy[0].delta:
                    enemy[3] -= enemy[0].delta
                    x_dist = waypoints[enemy[2]][0] - waypoints[enemy[2] - 1][0]
                    enemy[0].x += 0 if x_dist == 0 else x_dist / abs(x_dist)
                    y_dist = waypoints[enemy[2]][1] - waypoints[enemy[2] - 1][1]
                    enemy[0].y += 0 if y_dist == 0 else y_dist / abs(y_dist)

                    if enemy[0].x == waypoints[enemy[2]][0] and enemy[0].y == waypoints[enemy[2]][1]:
                        enemy[2] += 1
            elif enemy[2] == len(waypoints):
                self.enemies[i] = None
                base_elements += 1
        
        return base_elements, golds_earned

    def debug_init(self):
        with dpg.tree_node(label=self.wave_id):
            for i in range(len(self.enemies)):
                enemy = self.enemies[i]
                with dpg.tree_node(label=i):
                    dpg.add_text(f"Hp : {enemy[0].current_hp}/{enemy[0].max_hp}", tag=f"wave_{self.wave_id}_enemy_{i}")
    
    def debug_loop(self):
        for i in range(len(self.enemies)):
            enemy = self.enemies[i]
            if enemy is None:
                dpg.set_value(f"wave_{self.wave_id}_enemy_{i}", "Killed")
            elif enemy[2] > 0:
                dpg.set_value(f"wave_{self.wave_id}_enemy_{i}", f"Hp : {enemy[0].current_hp}/{enemy[0].max_hp}")

    def render(self, ground_size):
        for enemy in self.enemies:
            if enemy is None:
                continue
            if enemy[2] > 0:
                enemy[0].render(ground_size)
    
    def is_finished(self) -> bool:
        return self.enemies == [None] * len(self.enemies)

class Game:   
    def __init__(self, delta_time, hp, initial_golds, ground, ground_size, initial_towers, waves):
        self.delta_time = delta_time
        self.hp = hp
        self.golds = initial_golds
        self.ground: List[List[Ground]] = ground
        self.ground_size: int = ground_size
        self.towers: Union[List[Tower], None] = [tower if tower is not None else Towers.BASIC_TOWER(0, 0, 0) for tower in initial_towers]
        k = 0
        for i in range(len(self.ground)):
            for j in range(len(self.ground[i])):
                if self.ground[i][j] == Grounds.TOWER:
                    self.towers[k].x = i; self.towers[k].y = j
                    k += 1
        
        self.waves: List[Wave] = waves
        for i in range(len(self.waves)):
            self.waves[i].wave_id = i
        self.current_wave = 0

        self.waypoints: List[Tuple[int, int]] = []
        self.waypoints_progression: List[float] = []
        self.create_waypoints()
    
    def _is_in_grid(self, x, y) -> bool:
        return x >= 0 and x < len(self.ground) and y >= 0 and y < len(self.ground[0])
    
    def _find_start(self):
        for i in range(len(self.ground)):
            for j in range(len(self.ground[i])):
                if self.ground[i][j] == Grounds.START:
                    return [i, j]
        return [-1, -1]
    
    def _find_direction(self, start_point, previous_direction):
        for i in range(-1, 2):
            for j in range(-1, 2):
                cursor = [start_point[0] + i, start_point[1] + j]
                if not self._is_in_grid(cursor[0], cursor[1]):
                    continue
                if i == j or i == -j or (previous_direction is not None and [i, j] == [-previous_direction[0], -previous_direction[1]]):
                    continue
                if self.ground[start_point[0] + i][start_point[1] + j] == Grounds.PATH or self.ground[start_point[0] + i][start_point[1] + j] == Grounds.END:
                    return [i, j]
        return [0, 0]
    
    def create_waypoints(self):
        start_pos = self._find_start()
        if start_pos == [-1, -1]:
            raise "No ground start"
        
        self.waypoints = [start_pos]
        direction = None
        while True:
            direction = self._find_direction(self.waypoints[-1], direction)
            if direction == [0, 0]:
                raise "Not continuous path"
            cursor = [self.waypoints[-1][0] + 2 * direction[0], self.waypoints[-1][1] + 2 * direction[1]]
            while self._is_in_grid(cursor[0], cursor[1]) and self.ground[cursor[0]][cursor[1]] == Grounds.PATH:
                cursor[0] += direction[0]; cursor[1] += direction[1]
            self.waypoints.append([cursor[0] - direction[0], cursor[1] - direction[1]])

            if self.ground[self.waypoints[-1][0]][self.waypoints[-1][1]] == Grounds.END:
                break
        
        self.waypoints_progression = [0]
        for i in range(len(self.waypoints) - 1):
            self.waypoints_progression.append(self.waypoints_progression[-1] + abs(self.waypoints[i + 1][0] - self.waypoints[i][0]) + abs(self.waypoints[i + 1][1] - self.waypoints[i][1]))
        for i in range(len(self.waypoints_progression)):
            self.waypoints_progression[i] /= self.waypoints_progression[-1]
    
    def _render_ground(self):
        for y, row in enumerate(self.ground):
            for x, ground in enumerate(row):
                dpg.draw_rectangle((x * self.ground_size, y * self.ground_size),
                                ((x + 1) * self.ground_size, (y + 1) * self.ground_size),
                                color=ground.color,
                                fill=ground.color,
                                parent="drawing")

    def _render_waypoints(self):
        size = self.ground_size // 8
        for waypoint in self.waypoints:
            dpg.draw_rectangle(
                (waypoint[1] * self.ground_size + self.ground_size // 2 - size // 2, waypoint[0] * self.ground_size + self.ground_size // 2 - size // 2),
                (waypoint[1] * self.ground_size + self.ground_size // 2 + size // 2, waypoint[0] * self.ground_size + self.ground_size // 2 + size // 2),
                color=(150, 150, 150),
                fill=(150, 150, 150),
                parent="drawing"
            )

    def _render_enemies(self):
        self.waves[self.current_wave].render(self.ground_size)
    
    def _render_towers(self):
        for tower in self.towers:
            if tower.level > 0:
                tower.render(self.ground_size, range=tower.range)
        
    def render(self):
        dpg.delete_item("drawing", children_only=True)
        self._render_ground()
        self._render_waypoints()
        self._render_enemies()
        self._render_towers()
    
    def debug_init(self):
        with dpg.tree_node(label="Game", default_open=True):
            dpg.add_slider_int(label="Ground Size", default_value=game.ground_size, min_value=20, max_value=120, callback=game.update_ground_size)
            dpg.add_text(f"Hp : {game.hp}", tag="game_hp")
            dpg.add_text(f"Golds : {game.golds}", tag="game_golds")
            dpg.add_text(f"Current wave : {self.current_wave+1}/{len(self.waves)}", tag="game_current_wave")
            with dpg.tree_node(label="Waypoints"):
                for i in range(len(self.waypoints)):
                    dpg.add_text(f"{i}: {self.waypoints[i][0], self.waypoints[i][0]} ({round(100 * self.waypoints_progression[i])}%)")
            with dpg.tree_node(label="Waves"):
                for wave in self.waves:
                    wave.debug_init()
    
    def debug_loop(self):
        dpg.set_value("game_hp", f"Hp : {game.hp}")
        dpg.set_value("game_golds", f"Golds : {game.golds}")
        dpg.set_value("game_current_wave", f"Current wave : {self.current_wave+1}/{len(self.waves)}")
        self.waves[self.current_wave].debug_loop()
    
    def build_tower(self, id, tower_type) -> int:
        if self.towers[id].level > 0:
            return -1
        tower_price = TowersGlossary.TOWERS[tower_type](0, 0, 0).upgrade_price
        if self.golds < TowersGlossary.TOWERS[tower_type](0, 0, 0).upgrade_price:
            return tower_price - self.golds
        self.towers[id] = TowersGlossary.TOWERS[tower_type](1, self.towers[id].x, self.towers[id].y)
        return 0
    
    def upgrade_tower(self, id) -> int:
        if self.towers[id].level <= 0:
            return -1
        if self.golds < self.towers[id].upgrade_price:
            return self.towers[id].upgrade_price - self.golds
        self.golds -= self.towers[id].upgrade_price
        self.towers[id].upgrade()
        return 0
    
    def update_ground_size(self, sender, app_data, user_data):
        self.ground_size = app_data
        self.render()
    
    def start(self):
        self.waves[0].start()
    
    def step(self, action: "Action") -> bool:
        action_type, tower_id, tower_type = action.get_action()
        if action_type == Action.BUILD:
            golds_diff = self.build_tower(tower_id, tower_type)
            if golds_diff == -1:
                dpg.set_value("action_result", f"Already have tower id={tower_type} at {tower_id}")
            elif golds_diff == 0:
                dpg.set_value("action_result", f"Build tower id={tower_type} at {tower_id}")
            else:
                dpg.set_value("action_result", f"Missing {golds_diff} golds to build tower id={tower_type} at {tower_id}")
        elif action_type == Action.UPGRADE:
            golds_diff = self.upgrade_tower(tower_id)
            if golds_diff == -1:
                dpg.set_value("action_result", f"You need to build first tower at {tower_id}")
            elif golds_diff == 0:
                dpg.set_value("action_result", f"Success tower upgrade at {tower_id}")
            else:
                dpg.set_value("action_result", f"Missing {golds_diff} golds to upgrade tower at {tower_id}")
        
        if self.waves[self.current_wave].is_finished():           
            self.current_wave += 1
            if self.current_wave == len(self.waves):
                return False
            self.waves[self.current_wave].start()
        
        sorted_enemies = self.waves[self.current_wave].get_sorted_enemies(self.waypoints, self.waypoints_progression)
        for tower in self.towers:
            if tower.level <= 0:
                continue
            tower.update()
            for i in range(len(sorted_enemies) - 1, -1, -1):
                if sorted_enemies[i] is None:
                    break
                if tower.target(sorted_enemies[i][0]):
                    break
        
        enemies_at_end, golds_earned = self.waves[self.current_wave].update(self.waypoints)
        self.hp -= enemies_at_end
        self.golds += golds_earned
        self.render()
        if self.hp <= 0:
            return False
        return True

game = Game(
    0.15,
    10,
    15,
    [
        [Grounds.START, Grounds.PATH, Grounds.PATH, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE],
        [Grounds.NONE, Grounds.NONE, Grounds.PATH, Grounds.TOWER, Grounds.PATH, Grounds.PATH, Grounds.PATH],
        [Grounds.NONE, Grounds.TOWER, Grounds.PATH, Grounds.PATH, Grounds.PATH, Grounds.NONE, Grounds.PATH],
        [Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.TOWER, Grounds.PATH],
        [Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.PATH],
        [Grounds.NONE, Grounds.NONE, Grounds.PATH, Grounds.PATH, Grounds.PATH, Grounds.PATH, Grounds.PATH],
        [Grounds.NONE, Grounds.NONE, Grounds.PATH, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE],
        [Grounds.END, Grounds.PATH, Grounds.PATH, Grounds.NONE, Grounds.NONE, Grounds.NONE, Grounds.NONE],
    ],
    50,
    [
        Towers.BASIC_TOWER(1, 0, 0),
        None,
        None
    ],
    [
        Wave([
            (Enemies.BASIC_ENEMY(1), 2),
            (Enemies.BASIC_ENEMY(1), 8),
            (Enemies.BASIC_ENEMY(1), 20),
            (Enemies.BASIC_ENEMY(1), 25),
            (Enemies.BASIC_ENEMY(1), 35),
            (Enemies.BASIC_ENEMY(2), 50)
        ]),
        Wave([
            (Enemies.BASIC_ENEMY(1), 1),
            (Enemies.BASIC_ENEMY(1), 2),
            (Enemies.BASIC_ENEMY(1), 3),
            (Enemies.BASIC_ENEMY(1), 5),
            (Enemies.BASIC_ENEMY(1), 7),
            (Enemies.BASIC_ENEMY(3), 20),
            (Enemies.BASIC_ENEMY(5), 40),
        ]),
        Wave([
            (Enemies.BASIC_ENEMY(1), 1),
            (Enemies.BASIC_ENEMY(1), 2),
            (Enemies.BASIC_ENEMY(1), 3),
            (Enemies.BASIC_ENEMY(1), 5),
            (Enemies.BASIC_ENEMY(1), 10),
            (Enemies.BASIC_ENEMY(1), 15),
            (Enemies.BASIC_ENEMY(1), 20),
            (Enemies.BASIC_ENEMY(3), 35),
            (Enemies.BASIC_ENEMY(3), 40),
            (Enemies.BASIC_ENEMY(7), 60),
            (Enemies.BASIC_ENEMY(8), 70),
            (Enemies.BASIC_ENEMY(9), 80)
        ])
    ]
)

class Action:
    EMPTY = 0
    BUILD = 1
    UPGRADE = 2
    
    def __init__(self, level_towers_amount, tower_type_amount):
        self.build_space: int = level_towers_amount * tower_type_amount
        self.upgrade_space: int = level_towers_amount
        self.tower_type_amount = tower_type_amount
        self.action: List[int] = None
        self.reset()
    
    def reset(self):
        self.action = [1] + [0] * self.build_space + [0] * self.upgrade_space
    
    def set_empty(self):
        self.action = [0] + [0] * self.build_space + [0] * self.upgrade_space
    
    def is_reset(self):
        return self.action[0] == 1
    
    def get_action(self):
        if self.action[0] == 1:
            return Action.EMPTY, None, None
        build_actions = self.action[1:1+self.build_space]
        if sum(build_actions) == 1:
            choice = np.argmax(build_actions)
            return Action.BUILD, int(choice // self.tower_type_amount), int(choice % self.tower_type_amount)
        return Action.UPGRADE, int(np.argmax(self.action[1+self.build_space:])), None  

dpg.create_context()
dpg.create_viewport(title='Tower Defense Visualization', width=1280, height=720)
dpg.setup_dearpygui()

with dpg.window(label="Rendering Area", width=720, height=720, no_title_bar=True, no_scrollbar=True, no_resize=True):
    with dpg.drawlist(width=720, height=720, tag="drawing"):
        pass

dpg.show_viewport()
game.start()
landmark = time.time()

action = Action(len(game.towers), len(TowersGlossary.TOWERS))

def change_delta(sender, app_data, user_data):
    game.delta_time = app_data

instant_action = Action(len(game.towers), len(TowersGlossary.TOWERS))

def update_action(sender, app_data, user_data):
    instant_action.set_empty()
    instant_action.action[user_data] = int(app_data)
    for i in range(len(instant_action.action)):
        if i == user_data:
            continue
        dpg.set_value(f"checkbox_{i}", False)

def validate_action(sender, app_data, user_data):
    for i in range(len(instant_action.action)):
        action.action[i] = instant_action.action[i]
    instant_action.reset()
    for i in range(len(instant_action.action)):
        dpg.set_value(f"checkbox_{i}", True if instant_action.action[i] == 1 else False)
        

def debug_action():   
    with dpg.tree_node(label="Action input", default_open=True):
        dpg.add_checkbox(label="Do nothing", callback=update_action, user_data=0, default_value=True, tag="checkbox_0")
        with dpg.tree_node(label="Build", default_open=True):
            for i in range(len(game.towers)):
                with dpg.group(horizontal=True):
                    for j in range(len(TowersGlossary.TOWERS)):
                        box_id = i * len(TowersGlossary.TOWERS) + j + 1
                        dpg.add_checkbox(label=f"id={j} at {i} {game.towers[i].x, game.towers[i].y}", callback=update_action, user_data=box_id, tag=f"checkbox_{box_id}")
        with dpg.tree_node(label="Upgrade", default_open=True):
            for i in range(len(game.towers)):
                box_id = i + len(game.towers) * len(TowersGlossary.TOWERS) + 1
                dpg.add_checkbox(label=f"id={game.towers[i].id} at {i} {game.towers[i].x, game.towers[i].y}", callback=update_action, user_data=box_id, tag=f"checkbox_{box_id}")
        dpg.add_button(label="Execute", callback=validate_action)
        dpg.add_text(label="Action result: ", tag="action_result")

def ui():
    with dpg.window(label="Game", width=560, height=720, pos=(720, 0), no_title_bar=True, no_resize=True, no_scrollbar=True):
        dpg.add_slider_float(label="Delta time", default_value=game.delta_time, min_value=0.01, max_value=2.5, callback=change_delta)
        debug_action()
        game.debug_init()

ui()
while dpg.is_dearpygui_running():
    game.debug_loop()
    if time.time() - landmark > game.delta_time:
        if not game.step(action):
            break
        landmark += game.delta_time
        
        if not action.is_reset():
            action.reset()

    dpg.render_dearpygui_frame()

dpg.destroy_context()