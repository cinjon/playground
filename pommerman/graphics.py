from scipy.misc import imresize as resize
from datetime import datetime
from gym.utils import reraise
from random import randint
from time import strftime
import numpy as np
import math
import os

"""
'rendering' converts a display specification (such as :0) 
into an actual Display object.
Pyglet only supports multiple Displays on Linux.
"""
from gym.envs.classic_control import rendering

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`", suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

from . import utility
from . import constants

__location__ = os.path.dirname(os.path.realpath(__file__))
_resource_path = os.path.join(__location__, constants.RESOURCE_DIR)

layer_background = pyglet.graphics.OrderedGroup(0)
layer_foreground = pyglet.graphics.OrderedGroup(1)
layer_top = pyglet.graphics.OrderedGroup(2)

class Viewer(object):
    def __init__(self):
        self.window = None
        self.display = None
        self._agents = []
        self._agent_count = 0
        self._board_state = None
        self._batch = None
        self.window = None
        self._step = 0
        self._agent_view_size = 4
        self._is_partially_observable = False
        self.isopen = False

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def set_board(self, state):
        self._board_state = state
    
    def set_agents(self, agents):
        self._agents = agents
        self._agent_count = len(agents)

    def set_partially_observable(self, is_partially_observable):
        self._is_partially_observable = is_partially_observable

    def set_step(self, step):
        self._step = step

    def set_agent_view_size(self, view_size):
        self._agent_view_size = view_size

    def close(self):
        self.window.close()
        self.isopen = False

    def window_closed_by_user(self):
        self.isopen = False

    def save(self, path):
        now = datetime.now()
        filename = now.strftime('%m-%d-%y_%-H-%M-%S_') + str(self._step) + '.png'
        path = os.path.join(path, filename)
        pyglet.image.get_buffer_manager().get_color_buffer().save(path)


class PixelViewer(Viewer):
    def __init__(self, 
                display=None, 
                board_size=13, 
                agents=[], 
                partially_observable=False,
                game_type=None
        ):
        super().__init__()
        self.display = rendering.get_display(display)
        self._board_size = board_size
        self._agent_count = len(agents)
        self._agents = agents
        self._is_partially_observable = partially_observable

    def render(self):
        frames = self.build_frame()

        if self.window is None:
            height, width, _channels = frames.shape
            self.window = pyglet.window.Window(width=4*width, height=4*height, display=self.display, vsync=False, resizable=True)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = True
                
        assert len(frames.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(frames.shape[1], frames.shape[0], 'RGB', frames.tobytes(), pitch=frames.shape[1]*-3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()

    def build_frame(self):
        board = self._board_state
        board_size = self._board_size
        agents = self._agents
        human_factor = constants.HUMAN_FACTOR
        rgb_array = self.rgb_array(board, board_size, agents, self._is_partially_observable)

        all_img = resize(rgb_array[0], (board_size*human_factor, board_size*human_factor), interp='nearest')
        other_imgs = [
            resize(frame, (int(board_size*human_factor/4), int(board_size*human_factor/4)), interp='nearest')
            for frame in rgb_array[1:]
        ]

        other_imgs = np.concatenate(other_imgs, 0)
        img = np.concatenate([all_img, other_imgs], 1)

        return img


    @staticmethod
    def rgb_array(board, board_size, agents, is_partially_observable):
        agent_view_size = constants.AGENT_VIEW_SIZE
        frames = []

        all_frame = np.zeros((board_size, board_size, 3))
        num_items = len(constants.Item)
        for row in range(board_size):
            for col in range(board_size):
                value = board[row][col]
                if utility.position_is_agent(board, (row, col)):
                    num_agent = value - num_items
                    if agents[num_agent].is_alive:
                        all_frame[row][col] = constants.AGENT_COLORS[num_agent]
                else:
                    all_frame[row][col] = constants.ITEM_COLORS[value]

        all_frame = np.array(all_frame)
        frames.append(all_frame)

        for agent in agents:
            row, col = agent.position
            my_frame = all_frame.copy()
            for r in range(board_size):
                for c in range(board_size):
                    if is_partially_observable and not all([
                            row >= r - agent_view_size, row < r + agent_view_size,
                            col >= c - agent_view_size, col < c + agent_view_size
                    ]):
                        my_frame[r, c] = constants.ITEM_COLORS[constants.Item.Fog.value]
            frames.append(my_frame)

        return frames


class PommeViewer(Viewer):
    def __init__(self, 
                display=None, 
                board_size=13, 
                agents=[], 
                partially_observable=False,
                game_type=None
    ):
        super().__init__()
        self.display = rendering.get_display(display)
        board_height = constants.TILE_SIZE * board_size
        height =  math.ceil(board_height + (constants.BORDER_SIZE * 2) + (constants.MARGIN_SIZE * 3))
        width = math.ceil(board_height + board_height / 4 + (constants.BORDER_SIZE * 2) + constants.MARGIN_SIZE)

        self._height = height
        self._width = width
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.set_caption('Pommerman')
        self.isopen = True
        self._board_size = board_size
        self._resource_manager = ResourceManager()
        self._tile_size = constants.TILE_SIZE
        self._agent_tile_size =  (board_height / 4) / board_size
        self._agent_count = len(agents)
        self._agents = agents
        self._game_type = game_type
        self._is_partially_observable = partially_observable
        

        @self.window.event
        def close(self):
            self.window.close()
            self.isopen = False

    def render(self):
        self.window.switch_to()
        self.window.dispatch_events()
        self._batch = pyglet.graphics.Batch()

        background = self.render_background()
        text = self.render_text()
        agents = self.render_dead_alive()
        board = self.render_main_board()
        agents_board = self.render_agents_board()
        
        self._batch.draw()
        self.window.flip()

    def render_main_board(self):
        board = self._board_state
        size = self._tile_size
        x_offset = constants.BORDER_SIZE
        y_offset = constants.BORDER_SIZE
        top = self.board_top(-constants.BORDER_SIZE - 8)
        return self.render_board(board, x_offset, y_offset, size, top)

    def render_agents_board(self):
        x_offset = self._board_size * self._tile_size + constants.BORDER_SIZE + constants.MARGIN_SIZE
        size = self._agent_tile_size
        agents = []
        top = self._height - constants.BORDER_SIZE + constants.MARGIN_SIZE
        for agent in self._agents:
            y_offset = agent.agent_id * size * self._board_size + ( agent.agent_id * constants.MARGIN_SIZE ) + constants.BORDER_SIZE
            agent_board = self.agent_view(agent)
            sprite = self.render_board(agent_board, x_offset, y_offset, size, top)
            agents.append(sprite)
        return agents

    def render_board(self, board, x_offset, y_offset, size, top=0):
        sprites = []
        for r in range(self._board_size):
            for c in range(self._board_size):
                x = c * size + x_offset
                y = top - y_offset - r * size
                tile_state = board[r][c]
                tile = self._resource_manager.tile_from_state_value(tile_state)
                tile.width = size
                tile.height = size
                sprite = pyglet.sprite.Sprite(tile, x, y, batch=self._batch, group=layer_foreground)
                sprites.append(sprite)
        return sprites

    def agent_view(self, agent):
        if not self._is_partially_observable:
            return self._board_state

        agent_view_size = self._agent_view_size
        state = self._board_state
        fog_value = self._resource_manager.fog_value()
        row, col = agent.position
         
        for r in range(self._board_size):
            for c in range(self._board_size):
                if self._is_partially_observable and not all([
                            row >= r - agent_view_size, row < r + agent_view_size,
                            col >= c - agent_view_size, col < c + agent_view_size
                    ]):
                    state[r][c] = fog_value
        
        return state

    def render_background(self):
        image_pattern = pyglet.image.SolidColorImagePattern(color=constants.BACKGROUND_COLOR)
        image = image_pattern.create_image(self._width, self._height)
        return pyglet.sprite.Sprite(image, 0, 0, batch=self._batch, group=layer_background)

    def render_text(self):
        text = []
        board_top = self.board_top(y_offset=8)
        title_label = pyglet.text.Label('Pommerman', 
                    font_name='Cousine-Regular', 
                    font_size=36,
                    x=constants.BORDER_SIZE, 
                    y=board_top,
                    batch=self._batch,
                    group=layer_top)
        title_label.color = constants.TILE_COLOR
        text.append(title_label)

        info_text = ''
        if self._game_type is not None:
            info_text += 'Mode: ' + self._game_type.name + '   '

        info_text += 'Time: ' + strftime('%b %d, %Y %H:%M:%S')
        info_text += '   Step: ' + str(self._step)

        time_label = pyglet.text.Label(info_text, 
                    font_name='Arial', 
                    font_size=10,
                    x=constants.BORDER_SIZE, 
                    y=5,
                    batch=self._batch,
                    group=layer_top)
        time_label.color = constants.TEXT_COLOR
        text.append(time_label)
        return text

    def render_dead_alive(self):
        board_top = self.board_top(y_offset=5)
        image_size = 30
        spacing = 5
        dead = self._resource_manager.dead_marker()
        dead.width = image_size
        dead.height = image_size
        sprites = []

        for agent in self._agents:
            x = self.board_right(x_offset=-3) - (4 - agent.agent_id) * ( image_size + spacing)
            y = board_top 
            agent_image = self._resource_manager.agent_image(agent.agent_id)
            agent_image.width = image_size
            agent_image.height = image_size
            sprites.append(pyglet.sprite.Sprite(agent_image, x, y, batch=self._batch, group=layer_foreground))

            if agent.is_alive is False:
                sprites.append(pyglet.sprite.Sprite(dead, x, y, batch=self._batch, group=layer_top))

        return sprites

    def board_top(self, y_offset=0):
        return constants.BORDER_SIZE + (self._board_size * self._tile_size) + y_offset

    def board_right(self, x_offset):
        return constants.BORDER_SIZE + (self._board_size * self._tile_size) + x_offset


class ResourceManager(object):
    def __init__(self):
        self._index_resources()
        self._load_fonts()
        self.images = self._load_images()
        self._fog_value = self._get_fog_index_value()
    
    @staticmethod
    def _index_resources():
        # Tell pyglet where to find the resources
        pyglet.resource.path = [_resource_path]
        pyglet.resource.reindex()
 
    @staticmethod
    def _load_images():
        images_dict = constants.IMAGES_DICT
        for i in range(0, len(images_dict)):
            image_data = images_dict[i]
            image = pyglet.resource.image(image_data['file_name'])
            images_dict[i]['image'] = image
        
        return images_dict

    @staticmethod
    def _load_fonts():
        for i in range(0, len(constants.FONTS_FILE_NAMES)):
            font_path = os.path.join(_resource_path, constants.FONTS_FILE_NAMES[i])
            pyglet.font.add_file(font_path)
   
    @staticmethod
    def _get_fog_index_value():
        for id, data in constants.IMAGES_DICT.items():
            if data['name'] == 'Fog':
                return id

    def tile_from_state_value(self, value):
        img = self.images[value]
        return img['image']

    def agent_image(self, agentId):
        index = agentId + 16
        img = self.images[index]
        return img['image']

    def dead_marker(self):
        img = self.images[20]
        return img['image']

    def fog_value(self):
        return self._fog_value

    def fog_tile(self):
        img = self.images[self._fog_value]
        return img['image']
