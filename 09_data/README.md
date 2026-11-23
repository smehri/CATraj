# 09_data — downloaded data (not committed)

Everything in this folder except this file is ignored by git: the archives
are large and carry their own licences. The notebooks download what they need
when `DOWNLOAD = True` is set in their settings cell, and expect this layout:

```
09_data/
└── zenodo_1167595/                 # doi:10.5281/zenodo.1167595 (Ray et al., 2019)
    ├── P1_AIS_Data/                # nari_dynamic.csv, nari_static.csv, README_parameters.txt
    ├── P1_AIS_Codes/               # Ship Types List.csv, MMSI Country Codes.csv, ...
    ├── E1_ocean/                   # oc_february.csv (sea state), README_parameters.txt
    └── E2_Weather_Conditions/      # weather observations and stations
```

| used by | needs |
|---|---|
| `05_tutorials/01_trajectory_annotation.ipynb` (`SOURCE = "zenodo"`) | `P1_AIS_Data`, `P1_AIS_Codes`, `E2_Weather_Conditions` |
| `05_tutorials/02_cadp_compression.ipynb` | `P1_AIS_Data`, `E1_ocean` |
| `07_benchmarks/` | `P1_AIS_Data`, `E1_ocean` |

**Licences.** AIS: CC BY-NC-SA 4.0 (attribute NARI, Brest; no commercial use;
derived data under the same licence). Sea state (SHOM – IFREMER, WAVEWATCH
III): public, citation mandatory (Boudière et al., 2013,
[doi:10.1016/j.ijome.2013.11.010](https://doi.org/10.1016/j.ijome.2013.11.010)).
Weather (rp5.ua, Met Office / NOAA): public, cite the source.
