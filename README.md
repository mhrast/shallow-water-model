# Shallow Water Model

This Python program is able to solve the shallow water equations in conservative form in a channel around the Earth between 20°N and 70°N with a spatial resolution of 100 km (around one degree,
default 254x50 grid points). It uses the Lax-Wendroff method. The three prognostic variables are the two wind speeds $`u`$ and $`v`$, and the depth of the fluid layer $`h`$ above a constant
topology $`H`$.

The domain is periodic in the x direction and has solid north and south boundaries where $`v=0`$ and $`h`$ is fixed at its initial values. The domain size
is comparable to the perimeter of a sphere at a latitude of 45°, $`2\pi R_e \cos(45°)\cong`$28,000 km. For the implementation of a rotating Earth,
the beta-plane approximation is used. The results are displayed as an animation of the fluid depth $`h`$ and the relative vorticity $`\eta`$. The inline
animation can be paused with the space bar and viewed frame-by-frame with the arrow keys.

## Usage

Open the script `swe_solver.py` with your preferred Python IDE. You can adjust the values of gravity acceleration $`g`$, mean Coriolis parameter $`f`$,
meridional gradient of Coriolis parameter $`\beta`$, number of longitudinal/latitudinal gridpoints $`n_x`$, $`n_y`$, timestep $`dt`$, time between outputs and
the total simulation time. A switch for initial geostrophic balance or wind at rest and random noise in the initial height field is implemented as well.
For convenience, there is a second script `swe_solver_GUI.py` which uses a simple graphical user interface.


By default, there are nine initial conditions:
- [ ] `UNIFORM_WESTERLY`: initializes a westerly wind with a mean wind speed of 20 m/s
- [ ] `ZONAL_JET`: initializes an idealized Bickley jet, for which $`h \propto \tanh(y-\bar{y})`$
- [ ] `REANALYSIS`: uses a pre-defined potential height field from the ECMWF reanalysis from 1 July 2000 (matlab file `reanalysis.mat`)
- [ ] `GAUSSIAN_BLOB`: initializes a Gaussian-shaped wave on the left side of the domain
- [ ] `STEP`
- [ ] `CYCLONE_IN_WESTERLY`
- [ ] `SHARP_SHEAR`
- [ ] `EQUATORIAL_EASTERLY`: initializes an easterly wind that is proportional to a cosine
- [ ] `SINUSOIDAL`


By default, there are five possible orographies $`H`$:
- [ ] `FLAT`: sets the orography to zero everywhere
- [ ] `SLOPE`
- [ ] `GAUSSIAN_MOUNTAIN`: inserts a Gaussian-shaped mountain in the middle of the domain
- [ ] `EARTH_OROGRAPHY`: uses a pre-defined topology of Earth in the Northern Hemisphere (matlab file `digital_elevation_map.mat`)
- [ ] `SEA_MOUNT`: inserts a Gaussian-shaped sea mount that is 500 m below the surface at its highest point

## Dependencies

To run this program, you will need recent versions of the Python packages `numpy`, `scipy` and `matplotlib` besides the standard Python. Optionally, 
for opening the script `swe_solver_GUI.py` the package `PySimpleGUI` is required.

## Authors and acknowledgment
This model is customized and adapted after:


SHALLOW WATER MODEL
Copyright &copy; 2017 by Paul Connolly

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.

This model integrates the shallow water equations in conservative form
in a channel using the Lax-Wendroff method.  It can be used to
illustrate a number of meteorological phenomena.

## License
Copyright &copy; 2022 mhrast

Permission is hereby granted, free of charge, to any person obtaining a copy of this program and associated documentation files
(the "Program"), to deal in the Program without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Program, and to permit persons to whom the Program is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Program.

THE PROGRAM IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE PROGRAM OR THE USE OR OTHER DEALINGS IN THE PROGRAM.