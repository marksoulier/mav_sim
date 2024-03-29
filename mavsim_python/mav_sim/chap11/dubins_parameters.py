"""
 dubins_parameters
   - Dubins parameters that define path between two configurations

 mavsim_matlab
     - Beard & McLain, PUP, 2012
     - Update history:
         3/26/2019 - RWB
         4/2/2020 - RWB
         12/21 - GND
"""

from typing import cast

import numpy as np
from mav_sim.tools import types
from mav_sim.tools.rotations import rotz  # Function for rotation about z-axis
from mav_sim.tools.types import NP_MAT  # Type for a numpy matrix


class DubinsParamsStruct:
    """Class for passing out calculated parameters
    """

    __slots__ = [
        "L",
        "c_s",
        "lam_s",
        "c_e",
        "lam_e",
        "z1",
        "q1",
        "z2",
        "z3",
        "q3",
    ]
    def __init__(self) -> None:
        self.L: float           # path length
        self.c_s: types.NP_MAT  # starting circle center
        self.lam_s: int         # direction of the start circle (+1 for CW/right, -1 for CCW/left)
        self.c_e: types.NP_MAT  # ending circle center
        self.lam_e: int         # direction of the end circle (+1 for CW/right, -1 for CCW/left)
        self.z1: types.NP_MAT   # Point on halfspace 1 boundary
        self.q1: types.NP_MAT   # Normal vector for halfspace 1
        self.z2: types.NP_MAT   # Point on halfspace 2 boundary (note that normal vector is same as halfpace 1)
        self.z3: types.NP_MAT   # Point on halfspace 3 boundary
        self.q3: types.NP_MAT   # Normal vector for halfspace 3

    def print(self) -> None:
        """Print the commands to the console."""
        print(
            "\n=L",self.L,
            "\n=c_s",self.c_s,
            "\n=lam_s",self.lam_s,
            "\n=c_e",self.c_e,
            "\n=lam_e",self.lam_e,
            "\n=z1",self.z1,
            "\n=q1",self.q1,
            "\n=z2",self.z2,
            "\n=z3",self.z3,
            "\n=q3",self.q3,
        )

class DubinsPoints:
    """Struct for storing points and radius used for calculating Dubins path
    """
    __slots__ = [
        "p_s",
        "chi_s",
        "p_e",
        "chi_e",
        "radius",
    ]
    def __init__(self, p_s: NP_MAT, chi_s: float, p_e: NP_MAT, chi_e: float, radius: float) -> None:
        self.p_s = p_s          # starting position
        self.chi_s = chi_s      # starting course angle
        self.p_e = p_e          # ending position
        self.chi_e = chi_e      # ending course angle
        self.radius = radius    # radius of Dubin's paths arcs

    def extract(self) -> tuple[NP_MAT, float, NP_MAT, float, float]:
        """Extracts all of the elements into a tuple
        """
        return (self.p_s, self.chi_s, self.p_e, self.chi_e, self.radius)

class DubinsParameters:
    """Class for storing the parameters for a Dubin's path
    """
    def __init__(self, p_s: NP_MAT =9999*np.ones((3,1)), chi_s: float =9999,
                 p_e: NP_MAT =9999*np.ones((3,1)), chi_e: float =9999, R: float =9999) -> None:
        """ Initialize the Dubin's path

        Args:
            p_s: starting position
            chi_s: starting course angle
            p_e: ending position
            chi_e: ending course angle
            R: radius of Dubin's path arcs
        """
        # Store input parameters
        self.p_s = p_s      # starting position
        self.chi_s = chi_s  # starting course angle
        self.p_e = p_e      # ending position
        self.chi_e = chi_e  # ending course angle
        self.radius = R     # radius of Dubin's paths arcs

        # Initialize calculated parameters
        self.length: float          # Dubin's path length
        self.center_s: types.NP_MAT # starting circle center
        self.dir_s: float           # direction of the start circle (1 => "CW", and "CCW" otherwise)
        self.center_e: types.NP_MAT # ending circle center
        self.dir_e: float           # direction of the end circle (1 => "CW", and "CCW" otherwise)
        self.r1: types.NP_MAT       # Point on halfspace 1
        self.n1: types.NP_MAT       # Normal vector for halfspace 1
        self.r2: types.NP_MAT       # Point on halfspace 2 (note that normal vector is same as halfpace 1)
        self.r3: types.NP_MAT       # Point on halfspace 3
        self.n3: types.NP_MAT       # Normal vector for halfspace 3

        if R == 9999: # Infinite radius case - straight line
            dubin = DubinsParamsStruct()
            dubin.L = R
            dubin.c_s = p_s
            dubin.lam_s = 1
            dubin.c_e = p_s
            dubin.lam_e = 1
            dubin.z1 = p_s
            dubin.q1 = p_s
            dubin.z2 = p_s
            dubin.z3 = p_s
            dubin.q3 = p_s
        else:
            points = DubinsPoints(p_s=p_s, chi_s=chi_s, p_e=p_e, chi_e=chi_e, radius=R)
            dubin = compute_parameters(points)
        self.set(dubin)

    def set(self, vals: DubinsParamsStruct) -> None:
        """Sets the class variables based upon the Dubins parameter struct

        Args:
            vals: Values to be stored in the class
        """
        self.length = vals.L         # Dubin's path length
        self.center_s = vals.c_s     # starting circle center
        self.dir_s = vals.lam_s      # direction of the start circle (1 => "CW", and "CCW" otherwise)
        self.center_e = vals.c_e     # ending circle center
        self.dir_e = vals.lam_e      # direction of the end circle (1 => "CW", and "CCW" otherwise)
        self.r1 = vals.z1            # Point on halfspace 1
        self.n1 = vals.q1            # Normal vector for halfspace 1
        self.r2 = vals.z2            # Point on halfspace 2 (note that normal vector is same as halfpace 1)
        self.r3 = vals.z3            # Point on halfspace 3
        self.n3 = vals.q3            # Normal vector for halfspace 3

def compute_parameters(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculate the dubins paths parameters. Returns the parameters defining the shortest
       path between two oriented waypoints

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the shortest Dubins path
    """

    # Check to ensure sufficient distance between points
    (p_s, _, p_e, _, R) = points.extract()
    ell = np.linalg.norm(p_s[0:2] - p_e[0:2])
    if ell < 2 * R:
        raise ValueError('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()

    # Calculate distance and switching surfaces
    dubin.L = 99999.
    dubin.c_s = np.array([[0.], [0.],[0.]])
    dubin.lam_s = 1
    dubin.c_e = np.array([[0.], [0.],[0.]])
    dubin.lam_e = 1
    dubin.q1 = np.array([[0.], [0.],[0.]])
    dubin.z1 = np.array([[0.], [0.],[0.]])
    dubin.z2 = np.array([[0.], [0.],[0.]])
    dubin.z3 = np.array([[0.], [0.],[0.]])
    dubin.q3 = np.array([[0.], [0.],[0.]])

    return dubin

def calculate_rsr(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the right-straight-right case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    # Calculate distance and switching surfaces
    dubin.L = 99999.
    dubin.c_s = np.array([[0.], [0.],[0.]])
    dubin.lam_s = 1
    dubin.c_e = np.array([[0.], [0.],[0.]])
    dubin.lam_e = 1
    dubin.q1 = np.array([[0.], [0.],[0.]])
    dubin.z1 = np.array([[0.], [0.],[0.]])
    dubin.z2 = np.array([[0.], [0.],[0.]])
    dubin.z3 = np.array([[0.], [0.],[0.]])
    dubin.q3 = np.array([[0.], [0.],[0.]])

    return dubin

def calculate_rsl(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the right-straight-left case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    # Calculate distance and switching surfaces
    dubin.L = 99999.
    dubin.c_s = np.array([[0.], [0.],[0.]])
    dubin.lam_s = 1
    dubin.c_e = np.array([[0.], [0.],[0.]])
    dubin.lam_e = 1
    dubin.q1 = np.array([[0.], [0.],[0.]])
    dubin.z1 = np.array([[0.], [0.],[0.]])
    dubin.z2 = np.array([[0.], [0.],[0.]])
    dubin.z3 = np.array([[0.], [0.],[0.]])
    dubin.q3 = np.array([[0.], [0.],[0.]])

    return dubin

def calculate_lsr(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the left-straight-right case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    # Calculate distance and switching surfaces
    dubin.L = 99999.
    dubin.c_s = np.array([[0.], [0.],[0.]])
    dubin.lam_s = 1
    dubin.c_e = np.array([[0.], [0.],[0.]])
    dubin.lam_e = 1
    dubin.q1 = np.array([[0.], [0.],[0.]])
    dubin.z1 = np.array([[0.], [0.],[0.]])
    dubin.z2 = np.array([[0.], [0.],[0.]])
    dubin.z3 = np.array([[0.], [0.],[0.]])
    dubin.q3 = np.array([[0.], [0.],[0.]])

    return dubin

def calculate_lsl(points: DubinsPoints) -> DubinsParamsStruct:
    """Calculates the Dubins parameters for the left-straight-left case

    Args:
        points: Struct defining the oriented points and radius for the Dubins path

    Returns:
        dubin: variables for the Dubins path
    """

    # Initialize output and extract inputs
    dubin = DubinsParamsStruct()
    (p_s, chi_s, p_e, chi_e, R) = points.extract()

    # Calculate distance and switching surfaces
    dubin.L = 99999.
    dubin.c_s = np.array([[0.], [0.],[0.]])
    dubin.lam_s = 1
    dubin.c_e = np.array([[0.], [0.],[0.]])
    dubin.lam_e = 1
    dubin.q1 = np.array([[0.], [0.],[0.]])
    dubin.z1 = np.array([[0.], [0.],[0.]])
    dubin.z2 = np.array([[0.], [0.],[0.]])
    dubin.z3 = np.array([[0.], [0.],[0.]])
    dubin.q3 = np.array([[0.], [0.],[0.]])

    return dubin

def mod(x: float) -> float:
    """Computes the modulus of x with respect to 2 pi

    Args:
        x: Angle

    Returns:
        x: Angle modified to be between 0 and 2pi
    """
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x
