# Data licences

**The BSD 3-Clause licence in the repository root covers the code only.** The
data in this folder comes from third parties and carries its own terms, which
are stricter. Read them before redistributing or using this data.

Each dataset's own parameter file (`AIS_parameters.txt`,
`Weather_parameters.txt`) carries the full field descriptions and provenance.

---

## AIS — `AIS/`

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
  BSD grant on `02_src/` does not extend here.
- **SA** — anything derived from this data must be shared under the same
  licence. Annotated trajectories derived from it are derivative works.

---

## Weather — `context/Weather/`

Coastal weather observations.

| | |
|---|---|
| Source | [rp5.ua](http://rp5.ua/archive.php), sourced by the Met Office (United Kingdom) from data provided by NOAA (United States) |
| Licence | Public — **cite the source** |
| Coverage | 16 stations, longitude −10.00 to 0.00, latitude 45.00 to 51.00 |
| Full dataset | ~75,000 hourly observations, 2015-10-01 to 2016-03-31 UTC |
| Extract here | 400 observations, 2016-02-05 01:00 to 2016-02-06 01:00 UTC |

---

## Using your own data instead

Nothing in `catraj` depends on these files. They exist so the tutorial and the
app run without setup. Point either at your own trajectories and observations
and the licence questions here do not arise.
