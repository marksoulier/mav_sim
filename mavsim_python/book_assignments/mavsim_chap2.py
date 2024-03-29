"""
mavSimPy
    - Chapter 2 assignment for Beard & McLain, PUP, 2012
    - Update history:
        1/10/2019 - RWB
        12/2021 - GND
"""

import sys  # Used for exit

import mav_sim.parameters.simulation_parameters as SIM  # parameters
from mav_sim.chap2.mav_viewer import MavViewer  # Viewer for MAV
from mav_sim.message_types.msg_state import MsgState  # message types


def main() -> int:
    """Shows MAV moving

    Continually updates the transform variables

    Returns:
    0 if success
    """

    # initialize messages
    state = MsgState()  # instantiate state message

    # initialize viewers and video
    mav_view = MavViewer()

    # initialize the simulation time
    sim_time = SIM.start_time

    # main simulation loop
    while sim_time < SIM.end_time:
        # -------vary states to check viewer-------------
        if sim_time < SIM.end_time/6:
            state.north += 10*SIM.ts_simulation
        elif sim_time < 2*SIM.end_time/6:
            state.east += 10*SIM.ts_simulation
        elif sim_time < 3*SIM.end_time/6:
            state.altitude += 10*SIM.ts_simulation
        elif sim_time < 4*SIM.end_time/6:
            state.psi += 0.1*SIM.ts_simulation
        elif sim_time < 5*SIM.end_time/6:
            state.theta += 0.1*SIM.ts_simulation
        else:
            state.phi += 0.1*SIM.ts_simulation

        # -------update viewer and video-------------
        mav_view.update(state)

        # -------increment time-------------
        sim_time += SIM.ts_simulation

    print("Press Ctrl-Q to exit...")

    return 0

if __name__ == '__main__':
    sys.exit(main())
