"""path_follower.py implements a class for following a path with a mav
"""

import numpy as np
from mav_sim.chap2.transforms import rot_z
from mav_sim.message_types.msg_autopilot import MsgAutopilot
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools.wrap import wrap


class PathFollower:
    """Class for path following"""

    def __init__(self) -> None:
        """Initialize path following class"""
        self.chi_inf = np.radians(
            50
        )  # approach angle for large distance from straight-line path
        self.k_path = 0.01  # 0.05  # path gain for straight-line path following
        self.k_orbit = 1.0  # 10.0  # path gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path: MsgPath, state: MsgState) -> MsgAutopilot:
        """Update the control for following the path

        Args:
            path: path to be followed
            state: current state of the mav

        Returns:
            autopilot_commands: Commands to autopilot for following the path
        """
        if path.type == "line":
            self.autopilot_commands = follow_straight_line(
                path=path, state=state, k_path=self.k_path, chi_inf=self.chi_inf
            )
        elif path.type == "orbit":
            self.autopilot_commands = follow_orbit(
                path=path, state=state, k_orbit=self.k_orbit, gravity=self.gravity
            )
        return self.autopilot_commands


def follow_straight_line(
    path: MsgPath, state: MsgState, k_path: float, chi_inf: float
) -> MsgAutopilot:
    """Calculate the autopilot commands for following a straight line

    Args:
        path: straight-line path to be followed
        state: current state of the mav
        k_path: convergence gain for converging to the path
        chi_inf: Angle to take towards path when at an infinite distance from the path

    Returns:
        autopilot_commands: the commands required for executing the desired line
    """
    # Initialize the output
    autopilot_commands = MsgAutopilot()

    # set up variables from MsgPath
    q = path.line_direction
    r = path.line_origin
    airspeed = path.airspeed

    # define vector p pointing to plane
    p = np.array([[state.north], [state.east], [-state.altitude]])

    # calculate error vector
    e_p_i = p - r

    # qxn becomes [a2, -a1, 0]
    # calculate normal vecotor (before 10.5)
    n = np.array([[q.item(1)], [-q.item(0)], [0]]) / np.sqrt(
        q.item(0) ** 2 + q.item(1) ** 2
    )
    # s is vector of e_p projected onto the north down frame
    # s = a - (adotn)n
    s = e_p_i - np.dot(e_p_i.T, n) * n

    # calculate chi_q (angle of the path)
    # wrap 10.1 X_q
    X_q = np.arctan2(q.item(1), q.item(0))
    X_q = wrap(X_q, state.chi)

    # Define rotataion matrix
    R_i_p = rot_z(X_q)

    # calculate e_p (before 10.2)
    e_p = R_i_p @ (p - r)

    # error e_y
    # e_y = e_p.item(1)

    # equation 10.5 for height command
    autopilot_commands.altitude_command = -r[2][0] - (
        np.sqrt(s[0][0] ** 2 + s[1][0] ** 2)
        * q[2][0]
        / np.sqrt(q[0][0] ** 2 + q[1][0] ** 2)
    )

    # command airspeed
    autopilot_commands.airspeed_command = airspeed

    # feed forward will be zero
    autopilot_commands.phi_feedforward = 0.0

    # course command (10.8)
    autopilot_commands.course_command = X_q - chi_inf * 2 / np.pi * np.arctan(
        k_path * e_p.item(1)
    )

    return autopilot_commands


def follow_orbit(
    path: MsgPath, state: MsgState, k_orbit: float, gravity: float
) -> MsgAutopilot:
    """Calculate the autopilot commands for following a circular path

    Args:
        path: circular orbit to be followed
        state: current state of the mav
        k_orbit: Convergence gain for reducing error to orbit
        gravity: Gravity constant

    Returns:
        autopilot_commands: the commands required for executing the desired orbit
    """

    # Initialize the output
    autopilot_commands = MsgAutopilot()

    # set up variables from MsgPath
    c = path.orbit_center
    rho = path.orbit_radius
    orbit_direction = path.orbit_direction

    # calculate d (distance from orbit center to current position)
    d = np.sqrt((state.north - c.item(0)) ** 2 + (state.east - c.item(1)) ** 2)

    # calculate psi (angle from orbit center to current position)
    psi = np.arctan2(state.east - c.item(1), state.north - c.item(0))

    # wrap psi
    psi = wrap(psi, state.chi)

    # calculate chi_0 depending on the orbit direction
    if orbit_direction == "CW":
        lambda_ = 1
    else:
        lambda_ = -1

    # calculate chi (angle of the orbit)
    chi_0 = psi + lambda_ * np.pi / 2

    orbit_error = (d - rho) / rho
    # airspeed
    autopilot_commands.airspeed_command = path.airspeed
    autopilot_commands.altitude_command = -c.item(2)

    # course command (10.14)
    autopilot_commands.course_command = chi_0 + lambda_ * np.arctan(
        k_orbit * (d - rho) / rho
    )

    # phi feedforward command (10.15)
    if orbit_error < 10:
        autopilot_commands.phi_feedforward = lambda_ * np.arctan(
            (path.airspeed**2) / (gravity * rho)
        )
    else:
        autopilot_commands.phi_feedforward = 0.0

    return autopilot_commands
