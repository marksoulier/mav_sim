"""
compute_trim
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:
        12/29/2018 - RWB
        1/2022 - GND
"""

import numpy as np
from mav_sim.chap3.mav_dynamics_euler import (
    IND_EULER,
    derivatives_euler,
    euler_state_to_quat_state,
    quat_state_to_euler_state,
)
from mav_sim.chap4.mav_dynamics import forces_moments, update_velocity_data
from mav_sim.message_types.msg_delta import MsgDelta
from mav_sim.tools import types
from scipy.optimize import Bounds, minimize


def compute_trim(
    state0: types.DynamicState, Va: float, gamma: float, R: float = np.inf
) -> tuple[types.DynamicState, MsgDelta]:
    """Compute the trim equilibrium given the airspeed and flight path angle

    Args:
        state0: An initial guess at the state
        Va: air speed
        gamma: flight path angle
        R: radius - np.inf corresponds to a straight line

    Returns:
        trim_state: The resulting trim trajectory state
        trim_input: The resulting trim trajectory inputs
    """
    # Convert the state to euler representation
    state0_euler = quat_state_to_euler_state(state0)

    # Calculate the trim
    trim_state_euler, trim_input = compute_trim_euler(
        state0=state0_euler, Va=Va, gamma=gamma, R=R
    )

    # Convert and output the returned value
    trim_state = euler_state_to_quat_state(trim_state_euler)
    return trim_state, trim_input


def compute_trim_euler(
    state0: types.DynamicStateEuler, Va: float, gamma: float, R: float
) -> tuple[types.DynamicStateEuler, MsgDelta]:
    """Compute the trim equilibrium given the airspeed, flight path angle, and radius

    Args:
        state0: An initial guess at the state
        Va: air speed
        gamma: flight path angle
        R: radius - np.inf corresponds to a straight line

    Returns:
        trim_state: The resulting trim trajectory state
        trim_input: The resulting trim trajectory inputs
    """
    # define initial state and input
    delta0 = MsgDelta(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.5)
    x0 = np.concatenate((state0, delta0.to_array()), axis=0)

    # define equality constraints
    cons = {
        "type": "eq",
        "fun": lambda x: np.array(
            [
                # magnitude of velocity vector is Va
                velocity_constraint(x=x, Va_desired=Va),
            ]
        ),
        "jac": lambda x: np.array([velocity_constraint_partial(x=x)]),
    }
    # Define the bounds
    eps = 1e-12  # Small number to force equality constraint to be feasible during optimization (bug in scipy)
    lb, ub = variable_bounds(state0=state0, eps=eps)

    # solve the minimization problem to find the trim states and inputs
    psi_weight = 100000.0  # Weight on convergence of psi
    res = minimize(
        trim_objective_fun,
        x0[:, 0],
        method="SLSQP",
        args=(Va, gamma, R, psi_weight),
        bounds=Bounds(lb=lb, ub=ub),
        constraints=cons,
        options={"ftol": 1e-10, "disp": False},
    )

    # extract trim state and input and return
    trim_state = np.array([res.x[0:12]]).T
    trim_input = MsgDelta(
        elevator=res.x.item(12),
        aileron=res.x.item(13),
        rudder=res.x.item(14),
        throttle=res.x.item(15),
    )
    return trim_state, trim_input


def extract_state_input(x: types.NP_MAT) -> tuple[types.NP_MAT, MsgDelta]:
    """Extracts a state vector and control message from the aggregate vector

    Args:
        x: Euler state and inputs combined into a single vector

    Returns:
        states: Euler state vector
        delta: Control command
    """
    # Extract the state and input
    state = x[0:12]
    delta = MsgDelta(
        elevator=x.item(12), aileron=x.item(13), rudder=x.item(14), throttle=x.item(15)
    )
    return state, delta


def velocity_constraint(x: types.NP_MAT, Va_desired: float) -> float:
    """Returns the squared norm of the velocity vector - Va squared

    Args:
        x: Euler state and inputs combined into a single vector
            [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]
        Va_desired: Desired airspeed

    Returns:
        Va^2 - Va_desired^2
    """
    # Extract the state and input
    state, _ = extract_state_input(x)

    # Calculate va_2
    va_2 = state[3] ** 2 + state[4] ** 2 + state[5] ** 2

    return float(va_2 - Va_desired**2)


def velocity_constraint_partial(x: types.NP_MAT) -> list[float]:
    """Defines the partial of the velocity constraint with respect to x

    Args:
        x: Euler state and inputs combined into a single vector
            [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]

    Returns:
        16 element list containing the partial of the constraint wrt x
    """
    # Extract the state and input
    state, _ = extract_state_input(x)

    # ewtunr list od 16 0s
    partial = [0.0] * 16
    partial[3] = 2 * state.item(3)  # partial wrt u
    partial[4] = 2 * state.item(4)  # partial wrt v
    partial[5] = 2 * state.item(5)  # partial wrt w
    return partial


def variable_bounds(
    state0: types.DynamicStateEuler, eps: float
) -> tuple[list[float], list[float]]:
    """Define the upper and lower bounds of each the states and inputs as one vector.
       If an upper and lower bound is equivalent, then the upper bound is increased by eps to
       avoid a bug in scipy. If no bound exists then +/-np.inf is used.

    Each bound will be a list of the form
        [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]

    Args:
        state0: initial guess at the desired euler state
        eps: Small number (epsilon)

    Returns:
        lb: 16 element list defining the lower bound of each variable
        ub: 16 element list defining the upper bound of each variable
    """
    # -lower             pn                 pe                             pd
    lb = [
        state0.item(IND_EULER.NORTH),
        state0.item(IND_EULER.EAST),
        state0.item(IND_EULER.DOWN),
        #      u     v     w        phi       theta          psi
        -np.inf,
        0.0,
        -np.inf,
        -np.pi / 2,
        -np.pi / 2 + 0.1,
        state0.item(IND_EULER.PSI),
        #   p,  q,     r
        0.0,
        0.0,
        -np.inf,
        #    \delta_e   \delta_a  \delta_r   \delta_t
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        0.0,
    ]
    # -upper             pn                       pe                             pd
    ub = [
        state0.item(IND_EULER.NORTH) + eps,
        state0.item(IND_EULER.EAST) + eps,
        state0.item(IND_EULER.DOWN) + eps,
        #      u     v          w        phi       theta          psi
        np.inf,
        0.0 + eps,
        np.inf,
        np.pi / 2,
        np.pi / 2 - 0.1,
        state0.item(IND_EULER.PSI) + eps,
        #   p,      q,       r
        0.0 + eps,
        0.0 + eps,
        np.inf,
        #    \delta_e   \delta_a  \delta_r   \delta_t
        np.pi / 2,
        np.pi / 2,
        np.pi / 2,
        1.0,
    ]
    return lb, ub


def trim_objective_fun(
    x: types.NP_MAT, Va: float, gamma: float, R: float, psi_weight: float
) -> float:
    """Calculates the cost on the trim trajectory being optimized using an Euler state representation

    Objective is the norm of the desired dynamics subtract the actual dynamics (except the x-y position variables)

    Args:
        x: current state and inputs combined into a single vector
            [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]
        Va: relative wind vector magnitude
        gamma: flight path angle
        R: radius - np.inf corresponds to a straight line

    Returns:
        J: resulting cost of the current parameters
    """
    # Extract the state and input
    state, delta = extract_state_input(x)

    state_quat = euler_state_to_quat_state(state)

    # Calculate the desired trim trajectory dynamics
    trim_dynamics = np.zeros((12, 1))
    trim_dynamics[IND_EULER.DOWN] = -Va * np.sin(gamma)
    trim_dynamics[IND_EULER.PSI] = Va * np.cos(gamma) / R

    # Calculate forces
    Va, alpha, beta, _ = update_velocity_data(state=state_quat)
    forces = forces_moments(
        state=state_quat, delta=delta, Va=Va, alpha=alpha, beta=beta
    )

    # Calculate the dynamics based upon the current state and input (use euler derivatives)
    dynamics = derivatives_euler(state=state, forces_moments=forces)

    # Calculate the difference between the desired and actual
    diff = trim_dynamics - dynamics

    # Calculate the square of the difference (neglecting pn and pe)
    # declare Q matrix with 1s on the diagonal
    Q = np.eye(12)
    # change the first 2 diagnals to 0 and the 8th to psi_weight
    Q[0, 0] = 0
    Q[1, 1] = 0
    Q[8, 8] = psi_weight

    # matrix multiplication diff^T*Q*diff
    J = diff.T @ Q @ diff
    # extract the value
    J = J.item(0)

    return float(J)


# if __name__ == "__main__":
#     from mav_sim.chap3.mav_dynamics_euler import (
#         IND_EULER,
#         derivatives_euler,
#         euler_state_to_quat_state,
#         quat_state_to_euler_state,
#     )
#     from mav_sim.chap4.mav_dynamics import DynamicState

#     # Create the state
#     state = DynamicState()
#     # Create the trim state
#     Va_trim = 25.0
#     gamma_trim = 0.2
#     R = 800.0
#     trim_state, trim_input = compute_trim(state.convert_to_numpy(), Va_trim, gamma_trim)

#     # Print the results
#     print("Trim State:")
#     print(trim_state)
#     print("Trim Input:")
#     print(trim_input)
