-----------------------------------------------------------------------------------------------------
-- Object : Coastal Weather Observations
-- File name : table_weatherStation.csv.csv - table_windDirection.csv - table_wheatherObservation.csv
-- Source : rp5.ua (http://rp5.ua/archive.php) - sourced by Met Office (United Kingdom) based on data provided by NOAA (United States)
-- Dataset version : Downloaded 03/04/2017 
-- SRID : WGS84
-- Coverage : Longitude between -10.00 and 0.00 and Latitude between 45.00 and 51.00
-- Volume : 1 observation / hour, from 16 wheather stations => about 75000 rows in table_wheatherObservation
-- Period : 6 months (2015-10-01 00:00:00 UTC to 2016-03-31 23:59:59 UTC)
-- Licence : Public, cite the source
-- Description : Coastal weather observations from 16 stations located around Britany. The following describes fields included in the main file (observation. The two other file provide location of stations and description of wind condition
-----------------------------------------------------------------------------------------------------


Attribute		Data type		Description
-----------------------------------------------------------------------------------------------------

*  id_station		integer				id of the stations		
*  local_time 		integer				unix timestamp - number of seconds since "01/01/1970"
*  T 				double precision	Air temperature (°C) at 2 meter height above the earth's surface
*  Tn 				double precision	Minimum air temperature (°C) during the past period (not exceeding 12 hours).
*  Tx 				double precision	Maximum air temperature (°C) during the past period (not exceeding 12 hours).
*  P 				double precision	Atmospheric pressure reduced to mean sea level (mm of mercury)
*  U 				integer				Relative humidity (%) at a height of 2 meters above the earth's surface
*  id_windDirection 	integer			Mean wind direction (compass point) at a height of 10-12 meters above the earths surface over the 10-minute period immediately preceding the observation. 	
*  Ff 				integer				Mean wind speed at a height of 10-12 meters above the earths surface over the 10-minute period immediately preceding the observation (meters per seconds).
*  ff10 			integer				Maximum gust value at a height of 10-12 meters above the earths surface over the 10-minute period immediately preceding the observation (meters per seconds).
*  ff3 				integer				Maximum gust value at a height of 10-12 meters above the earths surface between the periods of observations (meters per seconds).
*  VV 				double precision	Horizontal visibility (km).
*  Td 				double precision: 	Dewpoint temperature at a height of 2 meters above the earth's surface (°C).
*  RRR 				double precision: 	Amount of precipitation (mm).
*  tR 				integer				The period of time during which the specified amount of precipitation was accumulated.
