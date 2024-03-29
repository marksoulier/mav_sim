"""plotter_args.py defines the argument structures for creating plotting windows
"""
from typing import Literal, Optional, Union


class PlotArgs:
    ''' Class for storing and validating StatePlot arguments

    Arguments:
        name (str): Name of the plot to be displayed in the legend.
            If None, the first state string is used for the plot name
        states (list of str): List of states that compose this plot.
            Note: Plot dimension is determined by the number of states in this list
        sigma_bounds (list of ints): adds the respective sigma bound plots
            Example: [1, 2, 3] --> plot 1-sigma, 2-sigma, and 3-sigma bounds around data
        is_angle (bool): If True, the data will be treated as an angle and will wrap between -pi and pi
        rad2deg (bool): If True, the data will be converted from radians to degrees.
            Note: If rad2deg is True, is_angle will be forced to True.
        max_length (int): Max number of data points to render
            Note: This is especially useful for 2D plots where the time_window param does not apply
        connect (bool): If False, the data will be represented by symbols, like a scatterplot,
            rather than lines.
        symbol (char): Character determining the symbol to be used.
            Example: 'o'=circle, 'd'=diamond, '+'=cross, 't'=triange, 's'=square
        symbol_size (float): Size of the symbols in units of the plot.
            Note: If px_mode is True, the symbol_size is given in pixels
        px_mode (bool): If True, symbols will maintain constant pixel size
        color (char): Character such as 'b', 'w', 'k', to set the color of the axis lines
            (see pyqtgraph.mkColor for valid options)
        hidden (bool): If True, this plot will not be rendered and will not show up in the legend
            This is useful for debugging or only viewing one of multiple plots in a plotbox
            with minimal changes to the initialization code
    '''
    def __init__(self, name: Optional[str] =None, states: Optional[list[str]] =None,
            sigma_bounds: Optional[list[int]] =None, is_angle: Optional[bool] =None, rad2deg: Optional[bool] =None,
            max_length: Optional[int] =None, connect: bool =True, symbol: Literal['o', 'd', '+', 't', 's'] ='o',
            symbol_size: float=2, px_mode: bool =True, color: Optional[str]=None, hidden: bool =False) -> None:
        # pylint: disable=too-many-arguments
        # Define name
        if name is not None:
            self.name = name
        elif states is not None:
            self.name = states[0]
        else:
            raise ValueError('Must provide a plot name or state names.')

        # Define states
        if states is not None:
            self.state_names = states
        else:
            self.state_names = [self.name]

        # Read in other args
        self.rad2deg: bool = False
        if rad2deg is not None:
            self.rad2deg = rad2deg
        self.is_angle: bool = False
        if is_angle is not None:
            self.is_angle = is_angle or self.rad2deg
        self.sigma_bounds = sigma_bounds
        self.max_length = max_length
        self.connect =  connect
        self.symbol = symbol
        self.symbol_size = symbol_size
        self.px_mode = px_mode
        self.color = color
        self.hidden = hidden

    def set_color(self, color: str) -> None:
        """Set color of plot
        """
        self.color = color

class PlotboxArgs:
    ''' Class for storing and validating StatePlotbox arguments

    Arguments:
        -title (str): title of the plotbox. If None, the first plot name will be used.
        -plots (list of PlotArgs or str objects): plots to be added to the plotbox
        -sigma_bounds (list of ints): adds the respective sigma bound plots
            Example: [1, 2, 3] --> plot 1-sigma, 2-sigma, and 3-sigma bounds around data
        -legend (bool): If True, a legend with plot names will be displayed
        -time_window (float): Amount of time in seconds to display along the x-axis for 1D plots
        -max_length (int): Max number of data points to render
            Note: This is especially useful for 2D plots where the time_window param does not apply
        -axis_color (char): Character such as 'b', 'w', 'k', to set the color of the axis lines
            (see pyqtgraph.mkColor for valid options)
        -axis_width (float): Width of axis lines
        -labels (dict): Sets the axis labels. Valid keys are 'left', 'right', 'bottom', and 'top'.
            Example: {'left':'x position  (m)', 'bottom':'time (s)'}
        -plot_hues (int): Number of distinct hues to use for multiple plots in the same plotbox
        -plot_min_hue (int): Beginning hue for the range of hues used for multiple plots.
        -plot_max_hue (int): Final hue for the range of hues used for multiple plots.
        -plot_min_value (int): Beginning value for the range of values used for multiple plots.
        -plot_max_value (int): Final value for the range of values used for multiple plots.
        -is_angle (bool): If True, the data will be treated as an angle and will wrap between -pi and pi
        -rad2deg (bool): If True, the data will be converted from radians to degrees.
            Note: If rad2deg is True, is_angle will be forced to True.
    '''
    def __init__(self, title: Optional[str] =None, plots: Optional[ Union[list[str], list[PlotArgs]] ] =None,
                 sigma_bounds: Optional[list[int]] =None,legend: bool=True, time_window: float =15.0,
                 max_length: Optional[int] =None, axis_color: str ='w', axis_width: float =1,
                 labels: Optional[dict[Literal['left', 'right', 'bottom', 'top'], str]] =None,
                 plot_hues: int =4, plot_min_hue: int =0, plot_max_hue: int =270, plot_min_value: int=200,
                 plot_max_value: int =255, is_angle: bool =False, rad2deg: bool =False) -> None:
        # pylint: disable=too-many-arguments, too-many-branches
        # Define title
        if title is not None:
            self.title = title
        elif plots is not None:
            if isinstance(plots[0], PlotArgs):
                self.title = plots[0].name
            elif isinstance(plots[0], str):
                self.title = plots[0]
            else:
                raise TypeError('plots input {} of incorrect type ({}). Expected PlotArgs or str object'.format(
                    plots[0], type(plots[0])))

        else:
            raise ValueError('Must provide a plotbox title or plot names.')

        # Read in plots
        self.plots = []
        if plots is not None:
            for p in plots:
                if isinstance(p, PlotArgs):
                    if p.sigma_bounds is None:
                        p.sigma_bounds = sigma_bounds
                    if p.max_length is None:
                        p.max_length = max_length
                    self.plots.append(p)
                elif isinstance(p, str):
                    self.plots.append(PlotArgs(p, sigma_bounds=sigma_bounds, max_length=max_length,
                    is_angle=is_angle, rad2deg=rad2deg))
                else:
                    raise TypeError('plots input {} of incorrect type ({}). Expected PlotArgs or str object'.format(p, type(p)))
        else:
            # If no plots are defined, assume the title and plot are the same
            self.plots.append(PlotArgs(self.title, sigma_bounds=sigma_bounds, max_length=max_length,
            is_angle=is_angle, rad2deg=rad2deg))

        # Save other params
        self.legend         = legend
        self.time_window    = time_window
        self.axis_color     = axis_color
        self.axis_width     = axis_width
        self.labels         = labels
        self.plot_hues      = max(plot_hues, len(self.plots))
        self.plot_min_hue   = plot_min_hue
        self.plot_max_hue   = plot_max_hue
        self.plot_min_value = plot_min_value
        self.plot_max_value = plot_max_value
