"""
mavsim_python
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
"""
import sys

import matplotlib.pyplot as plt
import numpy as np


class Signals:
    """Class capable of simulating a variety of signal types
    """
    def __init__(self,
                 amplitude: float = 1.0,
                 frequency: float = 1.0,
                 start_time: float = 0.0,
                 duration: float = 0.01,
                 dc_offset: float = 0.0) -> None:
        """Store variables to create a signal
        """
        self.amplitude: float = amplitude
        self.frequency = frequency  # Hz (1/sec)
        self.period = 1.0/frequency
        self.start_time = start_time  # sec
        self.duration = duration
        self.dc_offset = dc_offset
        self.last_switch = start_time

    def step(self, time: float) -> float:
        '''Step function'''
        if time >= self.start_time:
            y = self.amplitude
        else:
            y = 0.0
        return y + self.dc_offset

    def sinusoid(self, time: float) -> float:
        '''sinusoidal function'''
        if time >= self.start_time:
            y = self.amplitude*np.sin(self.frequency * time)
        else:
            y = 0.0
        return float(y + self.dc_offset)

    def square(self, time: float) -> float:
        '''square wave function'''
        if time < self.start_time:
            y = 0.0
        elif time < self.last_switch + self.period / 2.0:
            y = self.amplitude
        else:
            y = -self.amplitude
        if time >= self.last_switch + self.period:
            self.last_switch = time
        return y + self.dc_offset

    def sawtooth(self, time: float) -> float:
        '''sawtooth wave function'''
        if time < self.start_time:
            y = 0.0
        else:
            y = self.amplitude * (time-self.last_switch)
        if time >= self.last_switch + self.period:
            self.last_switch = time
        return y + self.dc_offset

    def trapezoid(self, time: float) -> float:
        '''trapezoidal wave function'''
        k = 0.075   # transition fraction (fraction of wave period of rise/fall: 0 to 0.25 )
        if time < self.start_time:
            y = 0.0
        elif time < (self.last_switch + k * self.period):
            y = self.amplitude * (time - self.last_switch) / (k * self.period)
        elif time < (self.last_switch + (0.5 - k) * self.period):
            y = self.amplitude
        elif time < (self.last_switch + (0.5 + k) * self.period):
            y = self.amplitude - self.amplitude * (time - (self.last_switch + (0.5 - k) * self.period)) / (k * self.period)
        elif time < (self.last_switch + (1 - k) * self.period):
            y = -self.amplitude
        else:
            y = -self.amplitude + self.amplitude * (time - (self.last_switch + (1 - k) * self.period)) / (k * self.period)
        if time >= self.last_switch + self.period:
            self.last_switch = time
        return y + self.dc_offset

    def polynomial(self, time: float) -> float:
        '''polynomial transition wave function'''
        k = 0.1   # transition fraction (fraction of wave period of rise/fall: 0 to 0.25 )
        tt = k * self.period    # transition time
        if time < self.start_time:
            y = 0.0
        elif time <= (self.start_time + tt):
            y = ((3.0 / (tt**2)) * self.amplitude * (time - self.start_time)**2
            - (2.0 / (tt**3)) * self.amplitude * (time - self.start_time)**3)
        elif time <= (self.last_switch + 0.5 * self.period - tt):
            y = self.amplitude
        elif time <= (self.last_switch + 0.5 * self.period + tt):
            y = (self.amplitude
            - (3.0 / (2.0 * tt)**2) * 2.0 * self.amplitude * (time - (self.last_switch + 0.5 * self.period - tt))**2
            + (2.0 / (2.0 * tt)**3) * 2.0 * self.amplitude * (time - (self.last_switch + 0.5 * self.period - tt))**3)
        elif time <= (self.last_switch + self.period - tt):
            y = -self.amplitude
        else:
            y = (-self.amplitude
            + (3.0 / (2.0 * tt)**2) * 2.0 * self.amplitude * (time - (self.last_switch + self.period - tt))**2
            - (2.0 / (2.0 * tt)**3) * 2.0 * self.amplitude * (time - (self.last_switch + self.period - tt))**3)
        if time >= self.last_switch + self.period + tt:
            self.last_switch = time - tt
        return y + self.dc_offset

    def impulse(self, time: float) -> float:
        '''impulse function'''
        if self.start_time <= time <= (self.start_time+self.duration):
            y = self.amplitude
        else:
            y = 0.0
        return y + self.dc_offset

    def doublet(self, time: float) -> float:
        '''doublet function'''
        if self.start_time <= time <= (self.start_time + self.duration):
            y = self.amplitude
        elif (self.start_time + self.duration) <= time <= (self.start_time + 2*self.duration):
            y = -self.amplitude
        else:
            y = 0.0
        return y + self.dc_offset

    def random(self, time: float) -> float:
        '''random function'''
        if time >= self.start_time:
            y = self.amplitude*np.random.randn()
        else:
            y = 0.0
        return y + self.dc_offset

def main() -> int:
    """Function for testing the signals
    """
    # instantiate the system
    input_signal = Signals(amplitude=2.0, frequency=0.25)
    Ts = 0.001

    # main simulation loop
    sim_time = -1.0
    time = [sim_time]
    #output = [input_signal.sinusoid(sim_time)]
    #output = [input_signal.step(sim_time)]
    #output = [input_signal.impulse(sim_time)]
    #output = [input_signal.doublet(sim_time)]
    #output = [input_signal.random(sim_time)]
    #output = [input_signal.square(sim_time)]
    # output = [input_signal.sawtooth(sim_time)]
    # output = [input_signal.trapezoid(sim_time)]
    output = [input_signal.polynomial(sim_time)]
    while sim_time <= 10.0:
        #y = input_signal.sinusoid(sim_time)
        #y = input_signal.step(sim_time)
        #y = input_signal.impulse(sim_time)
        #y = input_signal.doublet(sim_time)
        #y = input_signal.random(sim_time)
        #y = input_signal.square(sim_time)
        # y = input_signal.sawtooth(sim_time)
        # y = input_signal.trapezoid(sim_time)
        y = input_signal.polynomial(sim_time)
        sim_time += Ts   # increment the simulation time

        # update data for plotting
        time.append(sim_time)
        output.append(y)

    # plot output vs time
    plt.plot(time, output)
    plt.show()
    return 1

if __name__ == "__main__":
    sys.exit(main())
