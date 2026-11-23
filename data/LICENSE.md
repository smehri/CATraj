# Data licences

**The BSD 3-Clause licence in the repository root covers the code only.** The
data in this folder comes from third parties and carries its own terms, which
are stricter. Read them before redistributing or using this data.

This file covers the sample committed in `sample/`. Each dataset's own
parameter file (`AIS_parameters.txt`, `Weather_parameters.txt`,
`Coastline_parameters.txt`) carries the
full field descriptions and provenance.

The full archive you download into `zenodo_1167595/` comes from the same
source under the same terms, plus the sea state data. Its licences are listed
in section 7 of
[`docs/getting_the_data.md`](../docs/getting_the_data.md).

---

## AIS — `sample/AIS/`

Vessel positions from the Automatic Identification System.

| | |
|---|---|
| Source | Naval Academy, France ([ecole-navale.fr](https://www.ecole-navale.fr)) |
| Licence | **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** |
| Coverage | Longitude −10.00 to 0.00, latitude 45.00 to 51.00 |
| Full dataset | 18,648,556 decoded dynamic messages, 2015-10-01 to 2016-03-31 UTC |
| Extract here | 106,894 positions, 2016-02-05 01:00 to 2016-02-06 01:00 UTC |

**CC BY-NC-SA 4.0 means three things you must observe:**

- **BY** — attribute the Naval Academy, France.
- **NC** — **no commercial use.** This is stricter than the code licence: the
  BSD grant on `src/` does not extend here.
- **SA** — anything derived from this data must be shared under the same
  licence. Annotated trajectories derived from it are derivative works.

---

## Weather — `sample/Weather/`

Coastal weather observations.

| | |
|---|---|
| Source | [rp5.ua](http://rp5.ua/archive.php), sourced by the Met Office (United Kingdom) from data provided by NOAA (United States) |
| Licence | Public — **cite the source** |
| Coverage | 16 stations, longitude −10.00 to 0.00, latitude 45.00 to 51.00 |
| Full dataset | ~75,000 hourly observations, 2015-10-01 to 2016-03-31 UTC |
| Extract here | 400 observations, 2016-02-05 01:00 to 2016-02-06 01:00 UTC |

---

## Coastline — `sample/Coastline/`

Land polygons, used as barriers by CPLS compression.

| | |
|---|---|
| Source | European Environment Agency, "EEA coastline for analysis" (version 08/07/2013), obtained through the Zenodo record [doi:10.5281/zenodo.1167595](https://doi.org/10.5281/zenodo.1167595) |
| Licence | **EEA standard re-use policy**: re-use for commercial or non-commercial purposes is permitted free of charge, **provided the source is acknowledged**. Copyright holder: Directorate-General for Environment (DG ENV). See <http://www.eea.europa.eu/legal/copyright> |
| Coverage | Europe |
| Full dataset | 71,514 land polygons (`Europe Coastline (Polygone).shp`, EPSG:3035) |
| Extract here | 175 polygons, 12,049 vertices, clipped to the sample AIS area plus 5 km and reprojected to EPSG:32630 |

The rule here is attribution only: this layer is not under the AIS licence,
and the NC and SA terms above do not apply to it.

---

## Using your own data instead

Nothing in `catraj` depends on these files. They exist so the tutorial and the
app run without setup. Point either at your own trajectories and observations
and the licence questions here do not arise.
