-----------------------------------------------------------------------------------------------------
-- Object : Sea State Forecast
-- File name : [month].csv 
-- Source : SHOM - IFREMER
-- Dataset version : Downloaded 13/04/2017
-- SRID : WGS84
-- Coverage : Longitude between -10.00 and 0.00 and Latitude between 45.00 and 51.00
-- Volume : 6 forecast files - 1 file/1 month. 
-- Period : 6 months (2015-10-01 00:00:00 UTC to 2016-03-31 23:59:59 UTC)
-- Licence : Public. Usage subject to mandatory citation (below)
-- Description : Sea state forecast based on WAVEWATCH III model. The following describes fields included in the 6 files.
-----------------------------------------------------------------------------------------------------

Attribute	Data type				Description
-----------------------------------------------------------------------------------------------------

*  lat 		double precision        longitude in degrees (-180.0 to 180.0)
*  lon 		double precision        latitude in degrees (-90.0 to 90.0)
*  dpt 		double precision  		bottom depth in meters. (Undefined value=-16384)
*  wlv 		double precision 		sea surface height above sea level in meters (=>tidal effect). (Undefined value=-327.67)
*  hs 		double precision 		significant height of wind and swell waves (=>see state) (Undefined value=--65.534)
*  lm 		integer   				mean wave length in meters. (Undefined value=-32767)
*  dir 		double precision     	wave mean direction ("from direction") (Undefined value=-3276.7)
*  ts		bigint					unix timestamp - number of seconds since 1970/01/01
-----------------------------------------------------------------------------------------------------

REFERENCE 
Boudiere Edwige, Maisondieu Christophe, Ardhuin Fabrice, Accensi Mickael, Pineau-Guillou Lucia, Lepesqueur Jeremy (2013). A suitable metocean hindcast database for the design of Marine energy converters . International Journal of Marine Energy , 3-4, e40-e52 . http://doi.org/10.1016/j.ijome.2013.11.010