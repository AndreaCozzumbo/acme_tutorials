# import ligo.skymap.io.fits
# import ligo.skymap.postprocess.util
# import numpy as np
# import matplotlib.pyplot as pp
# from matplotlib.colors import LogNorm
# from matplotlib import cm
# import ligo.skymap.plot
# import astropy.io.fits as fits


# skymap , _ = ligo.skymap.io.fits.read_sky_map('GRB20251105_T03511_IPN_map_hpx.fits', nest = True) # NEST = True ensures that the order is rearranged properly, both for flat and multi-order maps

# print(skymap)

# ax = pp.axes(projection='astro degrees zoom',
#               center=f'278d 9d', radius='10 deg')

# cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap)

# vmax = np.percentile(skymap[~np.isnan(skymap)], 99.0)
# vmin = np.percentile(skymap[~np.isnan(skymap)], 10.0)
# vmin = max(vmax/1e3, vmin)

# skymap[cls > 100] = np.nan
# skymap[skymap == 0] = np.nan


# ax.imshow_hpx((skymap, 'ICRS'), nested=True, cmap=cm.Oranges, norm=LogNorm(vmin=vmin, vmax=vmax), zorder=0)
# ax.contour_hpx((cls, 'ICRS'), nested=True, colors='black', levels=(50, 90), zorder=1, linestyles=['dashed', 'solid'])

# pp.show()

import mhealpy as mhp
from astropy.table import QTable
import numpy as np
import matplotlib.pyplot as pp
from matplotlib.colors import LogNorm
from matplotlib import cm
import ligo.skymap.plot

# tab = QTable.read('GRB20251105_T03511_IPN_map_hpx_moc.fits')
# map = mhp.HealpixMap(data=tab['PROBDENSITY'], uniq=tab['UNIQ'])


# def moc_prob2percs(moc_skymap):

#     p_map = np.copy(moc_skymap.data)

#     inds_sort = np.argsort(p_map)[::-1]

#     perc_map = np.zeros_like(p_map)

#     perc_map[inds_sort] = np.cumsum((p_map*moc_skymap.pixarea())[inds_sort])

#     perc_map = mhp.HealpixMap(data=perc_map, uniq=moc_skymap.uniq)

#     return perc_map


# moc_perc_j = moc_prob2percs(map)


# vmax = np.percentile(map.data, 99.0)
# vmin = np.percentile(map.data[map.data>0], 10.0)
# vmin = max(vmax/1e3, vmin)

# im,ax = map.plot(norm=LogNorm(vmin=vmin, vmax=vmax), ax='astro degrees mollweide', cmap=cm.Oranges, rasterize=True, cbar=False)

# img_perc_j = moc_perc_j.get_wcs_img(ax, rasterize=False)

# CS = ax.contour(img_perc_j, levels=[0.9], colors=['black'], linestyles=['-'], linewidths=1.0)
# CS = ax.contour(img_perc_j, levels=[0.5], colors=['black'], linestyles=['--'], linewidths=1.0)

# ax.grid(True)

# pp.show()

# moc_skymap.uniq → array of UNIQ indices (int64)
# perc_map → data array

from astropy_healpix import nside_to_pixel_area
import astropy_healpix as ah 
import astropy.coordinates as coord
import astropy.units as u   
import healpy as hp


skymap = QTable.read('bayestar.multiorder.fits,1')
uniq = skymap['UNIQ']

level, ipix = ah.uniq_to_level_ipix(uniq)
nside = ah.level_to_nside(level)


# Nside → pixel area (steradians), vectorized
area_sr = nside_to_pixel_area(nside)

print(area_sr)

def moc_prob2percs(moc_skymap):

    p_map = np.copy(skymap['PROBDENSITY'])

    inds_sort = np.argsort(p_map)[::-1]

    perc_map = np.zeros_like(p_map)

    perc_map[inds_sort] = np.cumsum((p_map*area_sr.value)[inds_sort])

    q = QTable()

    q["PROBDENSITY"] = perc_map 
    q["UNIQ"] = skymap['UNIQ']

    return perc_map

moc_perc = moc_prob2percs(skymap)

# print credible level at ra = 278 deg, dec = 9 deg

ra = 180 * u.deg
dec = 55 * u.deg


max_level = 29
max_nside = ah.level_to_nside(max_level)
level, ipix = ah.uniq_to_level_ipix(skymap['UNIQ'])
index = ipix * (2**(max_level - level))**2

sorter = np.argsort(index)
match_ipix = ah.lonlat_to_healpix(ra, dec, max_nside, order='nested')
i = sorter[np.searchsorted(index, match_ipix, side='right', sorter=sorter)]

print("Credible level at ra,dec = ", moc_perc[i])

import ligo.skymap.io.fits
from ligo.skymap.postprocess.util import find_greedy_credible_levels
from astropy.coordinates import SkyCoord

skymap_arr, _ = ligo.skymap.io.fits.read_sky_map('bayestar.multiorder.fits,1', nest=True)
cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap_arr)

# Convert (RA, Dec) to HEALPix (theta, phi)
# theta = colatitude = π/2 - Dec (in radians)
# phi = RA (in radians)
ra_deg = 180
dec_deg = 55

theta = 0.5 * np.pi - np.deg2rad(dec_deg)  # Colatitude
phi = np.deg2rad(ra_deg)                    # Azimuth

# Query the multi-order map (handles variable resolution automatically)
ipix = hp.ang2pix(hp.get_nside(skymap_arr), theta, phi, nest=True)

print(f"Credible level at (RA={ra_deg}°, Dec={dec_deg}°): {cls[ipix]:.2f}%")