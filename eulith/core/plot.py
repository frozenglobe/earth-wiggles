# Plotting tools

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from utils import unit_vec_latlon, gcmax

def bounds(evpos:tuple[float], 
           stpos1:tuple[float], stpos2:tuple[float], 
           buffer:float=10.):
    """
    Calculate the bounding box for a map given event position and station pair positions. 

    :type evpos: tuple
    :param evpos: event position (lat, lon)
    :type stpos1: tuple
    :param stpos1: first station position (lat, lon)
    :type stpos2: tuple
    :param stpos2: second station position (lat, lon)
    :type buffer: float
    :param buffer: buffer in degrees to add to the bounding box
    :return: bounding box (lat_min, lat_max, lon_min, lon_max)
    """

    lats = [evpos[0], stpos1[0], stpos2[0]]
    lons = [evpos[1], stpos1[1], stpos2[1]]

    evv = unit_vec_latlon(lat=evpos[0], lon=evpos[1])
    for st in [stpos1, stpos2]:
        lon_pair = [st[1], evpos[1]]
        stv = unit_vec_latlon(lat=st[0], lon=st[1])
        gcmax_st = gcmax(evv, stv, 'sph', 'sph')
        max_latlon_st = (90 - gcmax_st[2] * 180 / np.pi, gcmax_st[1] * 180 / np.pi)
        if min(lon_pair) < max_latlon_st[1] < max(lon_pair): lats.append(max_latlon_st[0])
        else: continue

    lat_max = min(90, max(lats) + buffer)
    lat_min = max(-90, min(lats) - buffer)
    lon_max = min(180, max(lons) + buffer)
    lon_min = max(-180, min(lons) - buffer)

    return (lat_min, lat_max, lon_min, lon_max)
