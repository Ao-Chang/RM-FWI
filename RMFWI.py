"""
These are some helpers for the Random Mixing and wavelet_for_FWI
"""
from esys.escript import *
import numpy as np
from math import floor
def getMarmousi():
    """
    this return the Marmousi velocity as numpy array
    """
    VelocityFile="marmousi.bin"
    ShapeVelocityFile=(384, 122)

    f = open(VelocityFile,"rb")
    velocity_data = np.fromfile(f, dtype="f4").reshape(ShapeVelocityFile)
    return velocity_data[:,::-1]

def mapToDomain(domain, velocity_data, resolution, origin=(0.,0), v_top=1500):
    """
    this maps the velocity_data values into the domain and returns the velocity field as Data object.
    origin=(X0, Y0) is the position of the left lower vertex of the cell with value velocity_data[0,0].
    Values out side the region covered by velocity_data are extended from the boundary value except 
    for values above the top where the value v_top is used (water layer)).
    resolution gives the cell spacing of velocity_data 
    """
    NX, NZ=velocity_data.shape
    X0, Z0 =origin
    Width=resolution*NX
    Depth=resolution*NZ
    X=Function(domain).getX()
    
    v=Scalar(v_top, X.getFunctionSpace(), expanded=True)
    for p in range(v.getNumberOfDataPoints()):
        Xp = X.getTupleForDataPoint(p)
        x=Xp[0]
        z=Xp[1]
        if z < Z0:
            if x < X0:
                v_p=velocity_data[0,0]
            elif x < X0+Width:
                i=int(floor((x-X0)/resolution))
                v_p=velocity_data[i, 0]
            else:
                v_p=velocity_data[-1, 0]
        elif z < Z0+Depth:
            if x < X0:
                j=int(floor((z-Z0)/resolution))
                v_p=velocity_data[0, j]
            elif x < X0+Width:
                i=int(floor((x-X0)/resolution))
                j=int(floor((z-Z0)/resolution))
                v_p=velocity_data[i, j]
            else:
                j=int(floor((z-Z0)/resolution))
                v_p=velocity_data[-1, j]            
        else:
            v_p=v_top
        v.setValueOfDataPoint(p, float(v_p))
    v=interpolate(v, Function(domain))
    return v
