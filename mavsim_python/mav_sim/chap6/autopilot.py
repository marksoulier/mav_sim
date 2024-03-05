"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
        12/21 - GND
"""

import mav_sim.parameters.control_parameters as AP
import numpy as np
from mav_sim.chap6.pd_control_with_rate import PDControlWithRate
from mav_sim.chap6.pi_control import PIControl
from mav_sim.chap6.tf_control import TFControl
from mav_sim.message_types.msg_autopilot import MsgAutopilot
from mav_sim.message_types.msg_delta import MsgDelta
from mav_sim.message_types.msg_state import MsgState

# from mav_sim.tools.transfer_function import TransferFunction
from mav_sim.tools.wrap import saturate, wrap


class Autopilot:
    """Creates an autopilot for controlling the mav to desired values"""

    def __init__(self, ts_control: float) -> None:
        """Initialize the lateral and longitudinal controllers

        Args:
            ts_control: time step for the control
        """

        # instantiate lateral-directional controllers (note, these should be objects, not numbers)
        self.roll_from_aileron = PDControlWithRate(
            kp=AP.roll_kp, kd=AP.roll_kd, limit=np.radians(45)
        )
        self.course_from_roll = PIControl(
            kp=AP.course_kp, ki=AP.course_ki, limit=np.radians(30)
        )
        self.yaw_damper = TFControl(
            k=AP.yaw_damper_kr,
            n0=0,
            n1=1.0,
            d0=AP.yaw_damper_p_wo,
            d1=1.0,
            Ts=ts_control,
        )

        # instantiate longitudinal controllers (note, these should be objects, not numbers)
        self.pitch_from_elevator = PDControlWithRate(
            kp=AP.pitch_kp, kd=AP.pitch_kd, limit=np.radians(45)
        )
        self.altitude_from_pitch = PIControl(
            kp=AP.altitude_kp, ki=AP.altitude_ki, limit=np.radians(30)
        )
        self.airspeed_from_throttle = PIControl(
            kp=AP.airspeed_throttle_kp, ki=AP.airspeed_throttle_ki, limit=1.0
        )
        self.commanded_state = MsgState()

    def update(self, cmd: MsgAutopilot, state: MsgState) -> tuple[MsgDelta, MsgState]:
        """Given a state and autopilot command, compute the control to the mav

        Args:
            cmd: command to the autopilot
            state: current state of the mav

        Returns:
            delta: low-level flap commands
            commanded_state: the state being commanded
        """

        # lateral autopilot
        # command course angle
        command_course = wrap(chi_1=cmd.course_command, chi_2=state.chi)
        # commanded value for phi
        phi_c = cmd.phi_feedforward + self.course_from_roll.update(
            y_ref=command_course, y=state.chi
        )
        phi_c = saturate(phi_c, low_limit=np.radians(-30), up_limit=np.radians(30))
        delta_a = self.roll_from_aileron.update(y_ref=phi_c, y=state.phi, ydot=state.p)
        altitude_cmd = saturate(
            cmd.altitude_command,
            low_limit=state.altitude - AP.altitude_zone,
            up_limit=state.altitude + AP.altitude_zone,
        )
        delta_r = self.yaw_damper.update(y=state.r)
        theta_c = self.altitude_from_pitch.update(y_ref=altitude_cmd, y=state.altitude)

        # longitudinal autopilot
        delta_e = self.pitch_from_elevator.update(
            y_ref=theta_c, y=state.theta, ydot=state.q
        )
        delta_t = self.airspeed_from_throttle.update(
            y_ref=cmd.airspeed_command, y=state.Va
        )
        delta_t = saturate(delta_t, low_limit=0.0, up_limit=1.0)

        # construct control outputs and commanded states
        delta = MsgDelta(
            elevator=delta_e, aileron=delta_a, rudder=delta_r, throttle=delta_t
        )
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state.copy()
