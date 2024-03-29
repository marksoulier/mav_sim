"""
mavsim_python: world viewer (for chapter 12)
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - BGM
        12/21 - GND
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from mav_sim.chap2.draw_mav import DrawMav
from mav_sim.chap10.draw_path import DrawPath
from mav_sim.chap11.draw_waypoints import DrawWaypoints
from mav_sim.chap12.draw_map import DrawMap
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.message_types.msg_waypoints import MsgWaypoints
from mav_sim.message_types.msg_world_map import MsgWorldMap


class WorldViewer:
    """Object for viewing the world
    """
    def __init__(self) -> None:
        """Initialize viewer
        """
        self.scale = 4000
        # initialize Qt gui application and window
        self.app = pg.QtWidgets.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('World Viewer')
        self.window.setGeometry(0, 0, 1500, 1500)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(self.scale/20, self.scale/20, self.scale/20) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        self.window.setCameraPosition(distance=self.scale, elevation=50, azimuth=-90)
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_()  # bring window to the front
        self.plot_initialized = False  # has the mav been plotted yet?
        self.mav_plot: DrawMav
        self.path_plot: DrawPath
        self.waypoint_plot: DrawWaypoints
        self.map_plot: DrawMap

    def update(self, state: MsgState, path: MsgPath, waypoints: MsgWaypoints, world_map: MsgWorldMap) -> None:
        """Update the viewer given the mav state, path being followed, waypoints, and the map
        """
        blue = np.array([[30, 144, 255, 255]])/255.
        red = np.array([[1., 0., 0., 1]])
        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            self.map_plot = DrawMap(world_map, self.window)
            self.waypoint_plot = DrawWaypoints(waypoints, path.orbit_radius, blue, self.window)
            self.path_plot = DrawPath(path, red, self.window)
            self.mav_plot = DrawMav(state, self.window)
            self.plot_initialized = True
            path.plot_updated = True
            waypoints.plot_updated = True
        # else update drawing on all other calls to update()
        else:
            self.mav_plot.update(state)
            if not waypoints.plot_updated:
                self.waypoint_plot.update(waypoints)
                waypoints.plot_updated = True
            if not path.plot_updated:
                self.path_plot.update(path, red)
                path.plot_updated = True
        # redraw
        self.app.processEvents()
