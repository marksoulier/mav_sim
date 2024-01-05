"""ch4_derivatives_tests.py: Implements some basic tests for the chapter 4 martial."""


import os
import pickle
from typing import Any, cast

import numpy as np
from mav_sim.chap3.mav_dynamics import IND
from mav_sim.chap4.mav_dynamics import (
    forces_moments,
    gravitational_force,
    lateral_aerodynamics,
    longitudinal_aerodynamics,
    motor_thrust_torque,
    update_velocity_data,
)
from mav_sim.chap4.wind_simulation import WindSimulation
from mav_sim.message_types.msg_delta import MsgDelta
from mav_sim.tools import types


#############  Test structure definitions ########################
class GravitationalForceTest:
    """Stores a test for forces and moments"""
    def __init__(self) -> None:
        # Inputs
        self.quat: types.Quaternion

        # Outputs
        self.out: types.Vector

    def __str__(self) ->str:
        """Outputs the object in a nice format"""
        out =   "Inputs:\n quat:\n" + str(self.quat) + \
                "\n\nExpected result:\n" + str(self.out)

        return out

class LateralDynamicsTest:
    """Stores a test for forces and moments"""
    def __init__(self) -> None:
        # Inputs
        self.p: float
        self.r: float
        self.Va: float
        self.beta: float
        self.aileron: float
        self.rudder: float

        # Outputs
        self.out_f_lat: types.Vector
        self.out_torque_lat: types.Vector

    def __str__(self) ->str:
        """Outputs the object in a nice format"""
        out =   "Inputs:\n p: " + str(self.p) + \
                "\nr = " + str(self.r) +\
                "\nVa = " + str(self.Va) +\
                "\nbeta = " + str(self.beta) +\
                "\naileron = " + str(self.aileron) +\
                "\nrudder = " + str(self.rudder) +\
                "\n\nExpected results:\n f_lat:\n" + str(self.out_f_lat) +\
                "\ntorque_lat:\n" + str(self.out_torque_lat)

        return out

class LongitudinalDynamicsTest:
    """Stores a test for forces and moments"""
    def __init__(self) -> None:
        # Inputs
        self.q: float
        self.Va: float
        self.alpha: float
        self.elevator: float

        # Outputs
        self.out_f_lon: types.Vector
        self.out_torque_lon: types.Vector

    def __str__(self) ->str:
        """Outputs the object in a nice format"""
        out =   "Inputs:\n q: " + str(self.q) + \
                "\nVa = " + str(self.Va) +\
                "\nalpha = " + str(self.alpha) +\
                "\nelevator = " + str(self.elevator) +\
                "\n\nExpected results:\n f_lon:\n" + str(self.out_f_lon) +\
                "\ntorque_lon:\n" + str(self.out_torque_lon)

        return out

class ForcesMomentsTest:
    """Stores a test for forces and moments"""
    def __init__(self) -> None:
        # Inputs
        self.state: types.NP_MAT
        self.delta: MsgDelta = MsgDelta()
        self.Va: float
        self.beta: float
        self.alpha: float

        # Outputs
        self.out: types.ForceMoment

    def __str__(self) ->str:
        """Outputs the object in a nice format"""
        out =   "Inputs:\n state: " + str(self.state) + \
                "\ndelta.elevator = " + str(self.delta.elevator) +\
                "\ndelta.aileron = " + str(self.delta.aileron) +\
                "\ndelta.rudder = " + str(self.delta.rudder) +\
                "\ndelta.throttle = " + str(self.delta.throttle) +\
                "\nVa = " + str(self.Va) +\
                "\nbeta = " + str(self.beta) +\
                "\nalpha = " + str(self.alpha) +\
                "\n\nExpected result:\n" + str(self.out)

        return out

class MotorThrustTorqueTest:
    """Stores a test for motor_thrust_torque"""
    def __init__(self) -> None:
        # Inputs
        self.Va: float
        self.delta_t: float

        # Output
        self.T_p: float
        self.Q_p: float

    def __str__(self)->str:
        """Outputs string with inputs and outputs"""
        out =   "Inputs:\nVa: " + str(self.Va) + \
                "\ndelta_t: " + str(self.delta_t) + \
                "\nOutputs:\nT_p: " + str(self.T_p) + \
                "\nQ_p: " + str(self.Q_p)
        return out

class UpdateVelocityTest:
    """Stores a test for update_velocity_data"""
    def __init__(self) -> None:
        # Inputs
        self.state: types.NP_MAT
        self.wind: types.NP_MAT

        # Outputs
        self.Va: float
        self.alpha: float
        self.beta: float
        self.wind_inertial_frame: types.NP_MAT

    def __str__(self)->str:
        """Outputs the object in a nice format"""
        out =   "Inputs:\n state: " + str(self.state) + \
                "\nwind: " + str(self.wind) +\
                "\nOutputs: " + \
                "\nVa = " + str(self.Va) +\
                "\nbeta = " + str(self.beta) +\
                "\nalpha = " + str(self.alpha) +\
                "\wind_inertial_frame = " + str(self.wind_inertial_frame)

        return out

class WindSimulationTest:
    """Stores a test for the wind_simulation"""
    def __init__(self) -> None:
        # Inputs/Outputs
        self.updates: list[float] = []
        self.outputs: list[float] = []

    def __str__(self)->str:
        """Outputs the updates and outputs"""
        out = "Updates: ["
        for update in self.updates:
            out += str(update) + ", "
        out += "\nOutputs: ["
        for output in self.outputs:
            out += str(output) + ", "
        out += "]"
        return out


#############  Auto test generation ######################
def generate_tests()-> None:
    """Generates and saves a pickle file with the randomly generated tests"""
    # Generate tests
    data: dict[str,Any] = {}
    data["gravitational_force"] = generate_gravitational_force_tests()
    data["lateral_aerodyn"] = generate_lateral_aerodynamic_tests()
    data["longitudinal_aerodyn"] = generate_longitudinal_aerodynamic_tests()
    data["forces_moments"] = generate_forces_moments_tests()
    data["motor_cmd"] = generate_motor_thrust_torque_tests()
    data["update_vel"] = generate_update_velocity_data_tests()
    data["wind_simulation"] = generate_wind_simulation_tests()

    # Save the tests
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "ch04_test_archive.pkl"
        ),
        "wb",
    ) as file:
        pickle.dump(data, file)

def generate_gravitational_force_tests(num_tests: int = 100) -> list[GravitationalForceTest]:
    """Generates random test data and stores the results for the gravitational_force() function"""
    # Initailize the results structure
    tests: list[GravitationalForceTest] = []

    # Generate the results for each test
    for _ in range(num_tests):
        # Generate the state
        test = GravitationalForceTest()
        test.quat = np.random.randn(4,1)*100.
        test.quat = test.quat/np.linalg.norm(test.quat)

        # Generate the output
        test.out = gravitational_force(quat=test.quat)

        # Store the test
        tests.append(test)

    return tests

def generate_lateral_aerodynamic_tests(num_tests: int = 100) -> list[LateralDynamicsTest]:
    """Generates random test data and stores the results"""
    # Initailize the results structure
    tests: list[LateralDynamicsTest] = []

    # Generate the results for each test
    for k in range(num_tests):
        # Generate the state
        test = LateralDynamicsTest()
        test.p = np.random.randn()*100.
        test.r = np.random.randn()*100.

        # Generate the delta command
        test.aileron = np.random.randn()*np.pi/2
        test.rudder = np.random.randn()*np.pi/2

        # Generate wind parameters
        test.Va = np.random.rand()*35.
        test.beta = np.random.randn()*np.pi/2

        # test Va = 0
        if k == 0:
            test.Va = 0.

        # Generate the output
        test.out_f_lat, test.out_torque_lat = lateral_aerodynamics(
            p=test.p, r=test.r,
            Va=test.Va, beta=test.beta,
            aileron=test.aileron, rudder=test.rudder)

        # Store the test
        tests.append(test)

    return tests

def generate_longitudinal_aerodynamic_tests(num_tests: int = 100) -> list[LongitudinalDynamicsTest]:
    """Generates random test data and stores the results"""
    # Initailize the results structure
    tests: list[LongitudinalDynamicsTest] = []

    # Generate the results for each test
    for k in range(num_tests):
        # Generate the state
        test = LongitudinalDynamicsTest()
        test.q = np.random.randn()*100.

        # Generate the delta command
        test.elevator = np.random.randn()*np.pi/2

        # Generate wind parameters
        test.Va = np.random.rand()*35.
        test.alpha = np.random.randn()*np.pi/2

        # test Va = 0
        if k == 0:
            test.Va = 0.

        # Generate the output
        test.out_f_lon, test.out_torque_lon = longitudinal_aerodynamics(
            q=test.q,
            Va=test.Va,
            alpha=test.alpha,
            elevator=test.elevator
        )

        # Store the test
        tests.append(test)

    return tests


def generate_forces_moments_tests(num_tests: int = 100) -> list[ForcesMomentsTest]:
    """Generates random test data and stores the results"""
    # Initailize the results structure
    tests: list[ForcesMomentsTest] = []

    # Generate the results for each test
    for k in range(num_tests):
        # Generate the state
        test = ForcesMomentsTest()
        test.state = np.random.randn(13,1)*100.
        test.state[IND.QUAT] = test.state[IND.QUAT]/np.linalg.norm(test.state[IND.QUAT])

        # Generate the delta command
        test.delta.aileron = np.random.randn()*np.pi/2
        test.delta.elevator = np.random.randn()*np.pi/2
        test.delta.rudder = np.random.randn()*np.pi/2
        test.delta.throttle = np.random.rand()

        # Generate wind parameters
        test.Va = np.random.rand()*35.
        test.beta = np.random.randn()*np.pi/2
        test.alpha = np.random.randn()*np.pi/2

        # test Va = 0
        if k == 0:
            test.Va = 0.

        # Generate the output
        test.out = forces_moments(state=test.state, delta=test.delta, Va=test.Va, beta=test.beta, alpha=test.alpha)

        # Store the test
        tests.append(test)

    return tests

def generate_motor_thrust_torque_tests(num_tests: int = 100) -> list[MotorThrustTorqueTest]:
    """Generates random test data for and stores the results"""
    # Initailize the results structure
    tests: list[MotorThrustTorqueTest] = []

    # Generate the results for each test
    for k in range(num_tests):
        # Generate the inputs
        test = MotorThrustTorqueTest()
        test.Va = np.random.rand()*35.
        test.delta_t = np.random.rand()

        # test Va = 0
        if k == 0:
            test.Va = 0.

        # Generate the output
        test.T_p, test.Q_p = motor_thrust_torque(Va=test.Va, delta_t=test.delta_t)

        # Store the test
        tests.append(test)

    return tests

def generate_update_velocity_data_tests(num_tests: int = 100) -> list[UpdateVelocityTest]:
    """Generates random test data for update_velocity_data and stores the results"""
    # Initailize the results structure
    tests: list[UpdateVelocityTest] = []

    # Generate the results for each test
    for _ in range(num_tests):
        # Generate the state
        test = UpdateVelocityTest()
        test.state = np.random.randn(13,1)*100.
        test.state[IND.QUAT] = test.state[IND.QUAT]/np.linalg.norm(test.state[IND.QUAT])

        # Generate the wind
        test.wind = np.random.randn(6,1)*20.

        # Generate the output
        test.Va, test.alpha, test.beta, test.wind_inertial_frame = update_velocity_data(state=test.state, wind=cast(types.WindVector, test.wind) )

        # Store the test
        tests.append(test)

    return tests

def generate_wind_simulation_tests(num_tests: int = 100) -> list[WindSimulationTest]:
    """Generates random test data for WindSimulation and stores the results"""
    # Initailize the results structure
    tests: list[WindSimulationTest] = []

    # Generate the results for each test
    for test_num in range(num_tests*3):
        # Generate the state
        test = WindSimulationTest()
        wind_sim = WindSimulation(Ts=1.)

        # Generate the input / output date for the result
        for _ in range(10):
            # Get the update
            u = np.random.randn()
            test.updates.append(u)

            # Get the output
            output_num = test_num%3
            if output_num == 0:
                y = wind_sim.u_w.update(u)
            elif output_num == 1:
                y = wind_sim.v_w.update(u)
            else:
                y = wind_sim.w_w.update(u)
            test.outputs.append(y)

        # Store the test
        tests.append(test)

    return tests



#############  auto test run ######################
def gravitational_force_tests(tests: list[GravitationalForceTest], threshold: float = 1e-4) -> bool:
    """Runs the tests for gravitational_force()"""
    # Evaluate the results
    print("\nStarting gravitational_force test")
    success = True
    for test in tests:
        # Generate the output
        out = gravitational_force(quat=test.quat)

        # Compare the output with the result
        diff = np.linalg.norm(out - test.out)

        if diff > threshold:
            print("\n\nFailed test!")
            print("test:\n", test, "\nactual result:\n", out)
            success = False
            break

    # Indicate success
    if success:
        print("Passed test on gravitational_force")
    return success

def lateral_aerodynamics_tests(tests: list[LateralDynamicsTest], threshold: float = 1e-4) -> bool:
    """Runs the tests for lateral_aerodynamics()"""
    # Evaluate the results
    print("\nStarting lateral_aerodynamics test")
    success = True
    for test in tests:
        # Generate the output
        f_lat, torque_lat = lateral_aerodynamics(
            p=test.p, r=test.r,
            Va=test.Va, beta=test.beta,
            aileron=test.aileron, rudder=test.rudder)

        # Compare the output with the result
        diff = np.linalg.norm(f_lat - test.out_f_lat) + \
               np.linalg.norm(torque_lat - test.out_torque_lat)

        if diff > threshold:
            print("\n\nFailed test!")
            print("test:\n", test, "\nactual result: f_lat = \n", str(f_lat) )
            print("torque_lat = \n", str(torque_lat) )
            success = False
            break

    # Indicate success
    if success:
        print("Passed test on lateral_aerodynamics")
    return success

def longitudinal_aerodynamics_tests(tests: list[LongitudinalDynamicsTest], threshold: float = 1e-4) -> bool:
    """Runs the tests for longitudinal_aerodynamics()"""
    # Evaluate the results
    print("\nStarting longitudinal_aerodynamics test")
    success = True
    for test in tests:
        # Generate the output
        f_lon, torque_lon = longitudinal_aerodynamics(
            q=test.q,
            Va=test.Va,
            alpha=test.alpha,
            elevator=test.elevator)

        # Compare the output with the result
        diff = np.linalg.norm(f_lon - test.out_f_lon) + \
               np.linalg.norm(torque_lon - test.out_torque_lon)

        if diff > threshold:
            print("\n\nFailed test!")
            print("test:\n", test, "\nactual result: f_lon = \n", str(f_lon) )
            print("torque_lon = \n", str(torque_lon) )
            success = False
            break

    # Indicate success
    if success:
        print("Passed test on longitudinal_aerodynamics")
    return success


def forces_moments_tests(tests: list[ForcesMomentsTest], threshold: float = 1e-4) -> bool:
    """Runs the tests for forces_moments()"""
    # Evaluate the results
    print("\nStarting forces_moments test")
    success = True
    for test in tests:
        # Generate the output
        out = forces_moments(state=test.state, delta=test.delta, Va=test.Va, beta=test.beta, alpha=test.alpha)

        # Compare the output with the result
        diff = np.linalg.norm(out - test.out)

        if diff > threshold:
            print("\n\nFailed test!")
            print("test:\n", test, "\nactual result:\n", out)
            success = False
            break

    # Indicate success
    if success:
        print("Passed test on forces_moments")
    return success

def motor_thrust_torque_tests(tests: list[MotorThrustTorqueTest], threshold: float = 1e-4) -> bool:
    """Runs the tests for motor_thrust_torque()"""
    # Evaluate the results
    print("\nStarting motor_thrust_torque test")
    success = True
    for test in tests:
        # Generate the output
        T_p, Q_p = motor_thrust_torque(Va=test.Va, delta_t=test.delta_t)

        # Compare the output with the result
        diff = np.linalg.norm(T_p-test.T_p) + np.linalg.norm(Q_p-test.Q_p)

        if diff > threshold:
            print("\n\nFailed test!")
            print("test:\n", test, "\nactual T_p: ", T_p, "\nactual Q_p: ", Q_p)
            success = False
            break

    # Indicate success
    if success:
        print("Passed test on motor_thrust_torque")
    return success

def update_velocity_data_tests(tests: list[UpdateVelocityTest], threshold: float = 1e-4) -> bool:
    """Runs the tests for update_velocity_data()"""
    # Evaluate the results
    print("\nStarting update_velocity_data test")
    success = True
    for test in tests:
        # Generate the output
        Va, alpha, beta, wind_inertial_frame = update_velocity_data(state=test.state, wind=cast(types.WindVector, test.wind) )

        # Compare the output with the result
        diff = np.linalg.norm(Va-test.Va) + np.linalg.norm(alpha-test.alpha) + \
               np.linalg.norm(beta-test.beta) + np.linalg.norm(wind_inertial_frame-test.wind_inertial_frame)

        if diff > threshold:
            print("\n\nFailed test!")
            print("test:\n", test, "\nactual Va: ", Va, "\nactual alpha: ", alpha)
            print("actual beta: ", beta, "\nactual wind_inertial_frame:\n", wind_inertial_frame)
            success = False
            break

    # Indicate success
    if success:
        print("Passed test on update_velocity_data")
    return success

def wind_simulation_tests(tests: list[WindSimulationTest], threshold: float = 1e-4) -> bool:
    """Runs the tests for WindSimulation"""
    # Evaluate the results
    print("\nStarting WindSimulation test")
    success = True
    count = 0
    for test in tests:
        wind_sim = WindSimulation(Ts=1.)

        # Generate the input / output date for the result
        test_succ = True
        output_count = 0
        for in_out in zip(test.updates, test.outputs):
            # Extract the input/output
            u = in_out[0]
            output = in_out[1]

            # Get the output
            output_num = count%3
            if output_num == 0:
                y = wind_sim.u_w.update(u)
            elif output_num == 1:
                y = wind_sim.v_w.update(u)
            else:
                y = wind_sim.w_w.update(u)

            # Compare the output with the expected
            if np.linalg.norm(output-y) >= threshold:
                success = False
                print("\n\nFailed test!")
                print("test:\n", test, "\nActual_output[", output_count, "]: ", y)
                break

            # Increment the output counter
            output_count += 1

            if not success:
                break

        count += 1

    # Indicate success
    if success:
        print("Passed test on WindSimulation")
    return success

def run_auto_tests()->None:
    """Runs all of the auto-generated tests"""
    # Load the tests
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "ch04_test_archive.pkl"
        ),
        "rb",
    ) as file:
        data = pickle.load(file)

        # Run the tests
        succ = gravitational_force_tests(tests=data["gravitational_force"])
        if succ:
            succ = lateral_aerodynamics_tests(tests=data["lateral_aerodyn"])
        if succ:
            succ = longitudinal_aerodynamics_tests(tests=data["longitudinal_aerodyn"])
        if succ:
            succ = forces_moments_tests(tests=data["forces_moments"])
        if succ:
            succ = motor_thrust_torque_tests(tests=data["motor_cmd"])
        if succ:
            update_velocity_data_tests(tests=data["update_vel"])
        if succ:
            wind_simulation_tests(tests=data["wind_simulation"])

        if not succ:
            raise ValueError("Failed test")


def motor_thrust_torque_test() -> None:
    """Tests the motor_thrust_torque function."""
    print("Starting motor_thrust_torque test")
    # Inputs
    inputs = [
        {
            "Va": 1,
            "delta_t": 0,
        },
        {
            "Va": 27.39323489287441,
            "delta_t": 0.5,
        },
    ]
    # Expected outputs
    outputs = [
        (-0.033654137677838994, -0.002823682854958359),
        (-17.91352683604895, -0.7758235361365506),
    ]

    for input_it, output_it in zip(inputs, outputs):
        calculated_output = motor_thrust_torque(**input_it)  # type: ignore

        if (1e-6 < np.abs(np.array(calculated_output) - np.array(output_it))).any():
            print("Failed test!")
            print("Calculated output:")
            print(calculated_output)
            print("Expected output:")
            print(output_it)

    print("End of motor_thrust_torque test\n")

def forces_moments_test() -> None:
    """Tests the forces_moments function."""
    print("Starting forces_moments test")
    # Inputs
    inputs = [
        {
            "state": np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            "delta": MsgDelta(0, 0, 0, 0),
            "Va": 1,
            "beta": 0,
            "alpha": 0,
        },
        {
            "state": np.array(
                [
                    [24.283238643486627],
                    [12.605130052025968],
                    [1.2957327060769266],
                    [3.2299908091423797],
                    [0.2308339966235602],
                    [8.881532282520418],
                    [-0.025995661302161892],
                    [-0.011500703223228347],
                    [0.05851804333262313],
                    [0.10134276693843723],
                    [1.8427420637214973],
                    [5.2743652738342774],
                    [-0.5471458931221012],
                ]
            ),
            "delta": MsgDelta(-0.2, 0.001, 0.005, 0.5),
            "Va": 27.39323489287441,
            "beta": 0.022795289526122853,
            "alpha": 0.05259649205640062,
        },
    ]
    for input_it in inputs:
        input_it["state"][IND.QUAT] = input_it["state"][IND.QUAT] / np.linalg.norm(  # type: ignore
            input_it["state"][IND.QUAT]  # type: ignore
        )
    # Expected outputs
    outputs = [
        np.array(
            [
                [-3.40821628e-02],
                [0],
                [1.07829786e02],
                [2.82368285e-03],
                [8.94274083e-04],
                [0],
            ]
        ),
        np.array(
            [
                [-4.71982679],
                [87.12283471],
                [-113.4856833],
                [-44.44992726],
                [-31.38114459],
                [8.16544191],
            ]
        ),
    ]

    for input_it, output_it in zip(inputs, outputs):
        calculated_output = forces_moments(**input_it)  # type: ignore

        if (1e-6 < np.abs(calculated_output - output_it)).any():
            print("Failed test!")
            print("Calculated output:")
            print(calculated_output)
            print("Expected output:")
            print(output_it)

    print("End of forces_moments test\n")

def update_velocity_data_test() -> None:
    """Tests the update_velocity_data function."""
    print("Starting update_velocity_data test")
    # Inputs
    inputs = [
        {
            "state": np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            "wind": np.zeros([6, 1]),
        },
        {
            "state": np.array(
                [
                    [24.283238643486627],
                    [12.605130052025968],
                    [1.2957327060769266],
                    [3.2299908091423797],
                    [0.2308339966235602],
                    [8.881532282520418],
                    [-0.025995661302161892],
                    [-0.011500703223228347],
                    [0.05851804333262313],
                    [0.10134276693843723],
                    [1.8427420637214973],
                    [5.2743652738342774],
                    [-0.5471458931221012],
                ]
            ),
            "wind": np.array([ [1], [-2], [3], [-0.004], [0.005], [-0.006] ]),
        },
    ]
    for input_it in inputs:
        input_it["state"][IND.QUAT] = input_it["state"][IND.QUAT] / np.linalg.norm(
            input_it["state"][IND.QUAT]
        )
    # Expected outputs
    outputs = [
        (0, 0, 0, np.zeros([3, 1])),
        (
            10.379686170818202,
            1.252059888018447,
            -0.34301482267104366,
            np.array([ [1.00712983], [-2.00500799], [3.00104193] ]),
        ),
    ]

    for input_it, output_it in zip(inputs, outputs):
        calculated_output = update_velocity_data(**input_it)

        if (
            1e-6 < np.abs(np.array(calculated_output[0:2]) - np.array(output_it[0:2]))
        ).any() or (1e-6 < np.abs(calculated_output[3] - output_it[3])).any():
            print("Failed test!")
            print("Calculated output:")
            print(calculated_output)
            print("Expected output:")
            print(output_it)
    print("End of update_velocity_data test\n")

def wind_simulation_test() -> None:
    """Tests the WindSimulation class."""
    print("Starting WindSimulation test")
    # Inputs
    inputs = [
        {"u_w": 1, "v_w": 1, "w_w": 1},
        {"u_w": -0.5, "v_w": 0.35, "w_w": -0.89},
    ]
    # Expected outputs
    outputs = [
        np.array([0.00298834, 0.00365898, 0.00481975]),
        np.array([0.00149044, 0.00493313, 0.00049599]),
    ]

    test_class = WindSimulation(Ts=0.01, gust_flag=True)

    for input_it, output_it in zip(inputs, outputs):
        calculated_output = np.array(
            [
                test_class.u_w.update(input_it["u_w"]),  # type: ignore
                test_class.v_w.update(input_it["v_w"]),  # type: ignore
                test_class.w_w.update(input_it["w_w"]),  # type: ignore
            ]
        )

        if (1e-6 < np.abs(np.array(calculated_output) - np.array(output_it))).any():
            print("Failed test!")
            print("Calculated output:")
            print(calculated_output)
            print("Expected output:")
            print(output_it)

    print("End of WindSimulation test\n")

def run_all_tests() -> None:
    """Run all tests."""
    motor_thrust_torque_test()
    forces_moments_test()
    update_velocity_data_test()
    wind_simulation_test()
    run_auto_tests()


if __name__ == "__main__":
    #generate_tests()

    run_all_tests()
