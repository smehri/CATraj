# Getting the data

This page shows how to get every dataset the tutorials, workflows and benchmarks
use. Read it before you run anything beyond the 24-hour sample.

**In short.** Run this from the repository root:

```bash
python scripts/download_data.py
```

It downloads about 825 MB from Zenodo, checks each file against its published
checksum, and extracts what the code reads into `data/zenodo_1167595/`.
Running it again is safe: an interrupted download resumes, and files already
in place are skipped.

Then prepare the files the notebooks read (renames the coded columns, removes
noise; section 5, step 0):

```bash
python scripts/prepare_zenodo.py
```

---

## 1 · What needs what

| you want to run | data it needs | download |
|---|---|---|
| `notebooks/01_trajectory_annotation.ipynb` with the 24-hour sample (`SOURCE = "sample"`) | `data/sample/`, **already in the repository** | none |
| `app/` (the annotation app) | `data/sample/`, already in the repository | none |
| `scripts/benchmarks/equal_size_sample_data.py` | `data/sample/`, already in the repository | none |
| `notebooks/03_cpls_compression.ipynb` | `data/sample/` (AIS and coastline), already in the repository | none |
| `notebooks/01_trajectory_annotation.ipynb` with the full archive (`SOURCE = "zenodo"`) | AIS data, AIS codes, weather | `download_data.py annotation` (~203 MB) |
| `notebooks/02_cadp_compression.ipynb` | AIS data, ocean conditions | `download_data.py cadp` (~771 MB) |
| `scripts/benchmarks/prepare_feb2016.py`, `paper_experiments.py` | AIS data, ocean conditions | `download_data.py cadp` |
| `scripts/cadp_on_annotated_fleet.py` | the annotated fleet that tutorial 1 writes (section 5 below) | `download_data.py annotation` |
| the last section of `notebooks/03_cpls_compression.ipynb` (optional: the full coastline) | European coastline | `download_data.py cpls` (~53 MB) |

With `SOURCE = "auto"`, the default, tutorial 1 uses the full archive if it has
been downloaded and the sample otherwise. So it runs either way. The full
archive must also have been prepared with `scripts/prepare_zenodo.py` (section 5):
the notebooks read its cleaned files in `clean/`, not the raw CSVs.

---

## 2 · The source

Everything comes from one public Zenodo record:

> Ray, C., Dréo, R., Camossi, E., Jousselme, A.-L., Iphar, C. (2019). *Heterogeneous integrated dataset for Maritime Intelligence, Surveillance, and Reconnaissance.* Data in Brief 25, 104141. [doi:10.1016/j.dib.2019.104141](https://doi.org/10.1016/j.dib.2019.104141)
> Data: Zenodo, [doi:10.5281/zenodo.1167595](https://doi.org/10.5281/zenodo.1167595) · record <https://zenodo.org/records/1167595>

It covers 1 October 2015 to 31 March 2016, longitude −10° to 0° and latitude
45° to 51°: the western English Channel and the Bay of Biscay. The record has
19 archives. `catraj` uses five of them:

| Zenodo file | download | extracted, used here | MD5 (from the record) |
|---|---|---|---|
| [`[P1] AIS Data.zip`](https://zenodo.org/records/1167595/files/%5BP1%5D%20AIS%20Data.zip?download=1) | 201.4 MB | `nari_dynamic.csv` (1,125 MB, 19 million positions), `nari_static.csv` (86 MB, vessel type), `README_parameters.txt` | `90bec666a3c3474e18bb57676348b655` |
| [`[P1] AIS Status, Codes and Types.zip`](https://zenodo.org/records/1167595/files/%5BP1%5D%20AIS%20Status%2C%20Codes%20and%20Types.zip?download=1) | 8 kB | all 6 files; `Ship Types List.csv` names vessel types | `6f9bb328d6e657b16fea55210a6e56cf` |
| [`[E2] Weather Conditions.zip`](https://zenodo.org/records/1167595/files/%5BE2%5D%20Weather%20Conditions.zip?download=1) | 1.3 MB | all 5 files (6 MB); 16 coastal stations, hourly | `d8ac0644a670f584daaa4a450cf2baa7` |
| [`[E1] Ocean Conditions.zip`](https://zenodo.org/records/1167595/files/%5BE1%5D%20Ocean%20Conditions.zip?download=1) | 569.8 MB | `oc_february.csv` (675 MB; WAVEWATCH III sea state, 3-hourly, 43,206 sea points), `README_parameters.txt` | `20a1c8f8a6bbbd9bdf9821f938f22add` |
| [`[C2] European Coastline.zip`](https://zenodo.org/records/1167595/files/%5BC2%5D%20European%20Coastline.zip?download=1) | 52.6 MB | `Europe Coastline (Polygone).shp` with its `.shx`, `.dbf`, `.prj` (54 MB; 71,514 land polygons, EPSG:3035), `README_parameters.txt` | `52036af60c8427fcda4b867c706bf649` |

The ocean archive holds one file per month, `oc_october.csv` to
`oc_march.csv`, about 700 MB each. The tutorials use February 2016 only.

### Disk space

| | download | on disk once extracted |
|---|---|---|
| `annotation` | 203 MB | 1.2 GB |
| `cadp` | 771 MB | 1.9 GB |
| `cpls` | 53 MB | 54 MB |
| `all` | 825 MB | 2.0 GB |

The archives are kept by default (about 820 MB more), so nothing is downloaded
twice. Pass `--remove-zips` to delete each one after it is extracted. Running
the code adds more output: about 400 MB for `data/cache/cadp_feb2016.pkl` from the
benchmarks, and up to about 2 GB in `results/`.

---

## 3 · Three ways to get it

All three put the files in the same place (section 4). Pick one.

### A · The script (recommended)

```bash
conda activate catraj                               # any Python 3.10+ works; only the standard library is used
python scripts/download_data.py                # everything
python scripts/download_data.py annotation     # tutorial 1 only
python scripts/download_data.py cadp           # tutorial 2 and the benchmarks
python scripts/download_data.py cpls           # the full coastline for tutorial 3
python scripts/download_data.py --check        # report what is present, download nothing
```

| option | effect |
|---|---|
| `--months feb,jan` | sea-state months to extract: any of `oct nov dec jan feb mar`, or `all` (default `feb`) |
| `--remove-zips` | delete each archive after extracting it |
| `--check` | list what is present; exits with 1 if anything is missing |

It works from any folder, because paths are resolved from the script's own
location. `python scripts/download_data.py --help` lists the options.

### B · From inside a notebook

Both tutorials have a `DOWNLOAD` switch in their settings cell. Set
`DOWNLOAD = True` and run the cell. The notebook fetches the archives it needs
and verifies the same checksums. Set it back to `False` afterwards. The script
and the notebooks name the archives the same way, so whichever runs second
finds the files already there.

### C · By hand

Use this if Python cannot reach the internet, for example behind a firewall,
or if you want a download manager.

1. **Download** the archives you need from the table in section 2: click the
   links, or open the record at <https://zenodo.org/records/1167595>. From a
   terminal:

   ```bash
   # Linux, macOS, or Windows 10+ (in PowerShell type curl.exe, not curl).
   # -C - resumes an interrupted download; the URLs are percent-encoded on purpose.
   cd data
   mkdir zenodo_1167595
   cd zenodo_1167595
   curl -L -C - -o P1_AIS_Data.zip                    "https://zenodo.org/records/1167595/files/%5BP1%5D%20AIS%20Data.zip?download=1"
   curl -L -C - -o P1_AIS_Status_Codes_and_Types.zip  "https://zenodo.org/records/1167595/files/%5BP1%5D%20AIS%20Status%2C%20Codes%20and%20Types.zip?download=1"
   curl -L -C - -o E2_Weather_Conditions.zip          "https://zenodo.org/records/1167595/files/%5BE2%5D%20Weather%20Conditions.zip?download=1"
   curl -L -C - -o E1_Ocean_Conditions.zip            "https://zenodo.org/records/1167595/files/%5BE1%5D%20Ocean%20Conditions.zip?download=1"
   curl -L -C - -o C2_European_Coastline.zip          "https://zenodo.org/records/1167595/files/%5BC2%5D%20European%20Coastline.zip?download=1"
   ```

2. **Verify** each file against the MD5 in section 2. A mismatch means the
   download is incomplete or corrupt: delete it and download it again.

   ```bash
   md5sum *.zip                                   # Linux
   md5 *.zip                                      # macOS
   Get-FileHash *.zip -Algorithm MD5              # Windows PowerShell
   ```

3. **Extract** into the folders the code expects (section 4). Extract only
   the month you need from the ocean archive: it holds 4.2 GB in all.

   ```bash
   # Linux / macOS (unzip)
   unzip -n P1_AIS_Data.zip nari_dynamic.csv nari_static.csv README_parameters.txt -d P1_AIS_Data
   unzip -n P1_AIS_Status_Codes_and_Types.zip -d P1_AIS_Codes
   unzip -n E2_Weather_Conditions.zip -d E2_Weather_Conditions
   unzip -n E1_Ocean_Conditions.zip oc_february.csv README_parameters.txt -d E1_ocean
   unzip -n C2_European_Coastline.zip "Europe Coastline (Polygone).*" README_parameters.txt -d C2_European_Coastline
   ```

   ```powershell
   # Windows 10+ (tar reads zip files; create the folders first)
   mkdir P1_AIS_Data, P1_AIS_Codes, E2_Weather_Conditions, E1_ocean, C2_European_Coastline
   tar -xf P1_AIS_Data.zip -C P1_AIS_Data nari_dynamic.csv nari_static.csv README_parameters.txt
   tar -xf P1_AIS_Status_Codes_and_Types.zip -C P1_AIS_Codes
   tar -xf E2_Weather_Conditions.zip -C E2_Weather_Conditions
   tar -xf E1_Ocean_Conditions.zip -C E1_ocean oc_february.csv README_parameters.txt
   tar -xf C2_European_Coastline.zip -C C2_European_Coastline "Europe Coastline (Polygone).shp" "Europe Coastline (Polygone).shx" "Europe Coastline (Polygone).dbf" "Europe Coastline (Polygone).prj" README_parameters.txt
   ```

4. **Check** that it is all in place:
   `python scripts/download_data.py --check`.

---

## 4 · Where the files go

Paths are relative to the repository root. The code looks only here, and git
ignores everything in `data/` except its README, its LICENSE and the
committed 24-hour sample in `sample/`.

```
data/
├── sample/                           the sample (AIS, Weather, Coastline), committed (not used here)
├── cache/                            files the benchmarks build, e.g. cadp_feb2016.pkl
└── zenodo_1167595/
    ├── P1_AIS_Data/
    │   ├── nari_dynamic.csv          AIS positions             tutorials 1, 2 · benchmarks
    │   ├── nari_static.csv           vessel static data        tutorial 1
    │   └── README_parameters.txt     field descriptions        tutorial 1
    ├── P1_AIS_Codes/
    │   └── Ship Types List.csv       ship type code -> name    tutorial 1   (+ 5 other code tables)
    ├── E2_Weather_Conditions/
    │   ├── table_wheatherObservation.csv   hourly observations (the misspelling is the archive's)
    │   ├── table_weatherStation.csv        station positions
    │   └── README_parameters.txt           field descriptions        tutorial 1
    ├── E1_ocean/
    │   ├── oc_february.csv           sea state, Feb 2016       tutorial 2 · benchmarks
    │   └── README_parameters.txt
    ├── C2_European_Coastline/
    │   ├── Europe Coastline (Polygone).shp   (+ .shx .dbf .prj)  land polygons   tutorial 3 (optional section)
    │   └── README_parameters.txt
    └── clean/                        written by scripts/prepare_zenodo.py (section 5)
        ├── ais_dynamic.parquet       AIS, descriptive names, filtered     tutorials 1, 2
        ├── ocean_february.parquet    sea state per month (ocean_<month>)  tutorial 2
        └── weather.parquet           observations + station positions     tutorial 1
```

---

## 5 · After downloading: what to run, in order

| step | command | reads | writes |
|---|---|---|---|
| 0 | `python scripts/prepare_zenodo.py` (needs `pyarrow`) | the raw CSVs from the download | `data/zenodo_1167595/clean/*.parquet`: columns renamed, "not available" codes as NaN, noise removed |
| 1 | open `notebooks/01_trajectory_annotation.ipynb`, keep `SOURCE = "auto"` (or `"zenodo"`), run all | `clean/ais_dynamic.parquet`, `clean/weather.parquet`, the README files | `results/01_trajectory_annotation/`, including `06_fleet_annotated.parquet` (~1 GB) when `EXPORT_FLEET = True` |
| 2 | `python scripts/cadp_on_annotated_fleet.py` | `06_fleet_annotated.parquet` from step 1 | `results/02_cadp_compression/` |
| 3 | open `notebooks/02_cadp_compression.ipynb`, run all | `clean/ais_dynamic.parquet`, `clean/ocean_february.parquet` | figures and tables inline |
| 4 | `python scripts/benchmarks/prepare_feb2016.py`, then `python scripts/benchmarks/paper_experiments.py` | the same two files | `data/cache/cadp_feb2016.pkl`, then the paper's experiments |

Steps 1 and 3 depend on step 0; step 2 depends on step 1. Step 4 reads the raw
CSVs and depends on none of them.

Step 0 skips outputs that exist (`--force` rebuilds them), skips datasets that
were not downloaded, and takes a few minutes for the full AIS file. It median-filters
over `--window` seconds (default 30): this changes the AIS positions, speed,
course and heading, and leaves the hourly weather and the 3-hourly sea state as
they are, so every column the notebooks use goes through the same step.

Tutorial 1 reads the whole archive by default (`WINDOW` covers all six months,
and `MAX_VESSELS = None` takes every vessel). Loading `nari_dynamic.csv` in
full needs several GB of memory. On a smaller machine, narrow `WINDOW` to a
few days or set `MAX_VESSELS`. Step 2 then processes whatever tutorial 1
exported.

---

## 6 · Troubleshooting

| symptom | what to do |
|---|---|
| the download stops partway, or the connection drops | run the script again: it resumes from the `.part` file |
| `checksum mismatch` | the archive was deleted; run again to download it afresh. If it happens again, download it by hand (3 C) |
| very slow, or blocked by a firewall | the script honours the `HTTPS_PROXY` environment variable; otherwise download by hand with a browser or a download manager (3 C) |
| Zenodo returns an error, e.g. 429 *Too Many Requests* or 5xx | wait a few minutes and run the script again |
| a notebook says the cleaned files are missing | run `python scripts/prepare_zenodo.py` (after the download) |
| `prepare_zenodo.py` says pyarrow is needed | `pip install pyarrow` (or `pip install -e .[data]`) |
| a notebook raises `FileNotFoundError: missing: data/...` | the file is not where section 4 says; run `python scripts/download_data.py --check` |
| `scripts/cadp_on_annotated_fleet.py` says the Parquet file is missing | run tutorial 1 with the full archive and `EXPORT_FLEET = True` first (section 5) |
| out of disk space | extract only what you need (`annotation` or `cadp`), add `--remove-zips`, and keep `--months feb` |

---

## 7 · Licences and citation

The data is **not** covered by `catraj`'s BSD licence. Each part has its own
terms.

* **AIS data** ([P1]): **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.
  *BY*: attribute the Naval Academy Research Institute (NARI), Brest, and cite
  the dataset. *NC*: **no commercial use**. *SA*: anything derived from it,
  including annotated or compressed trajectories, results and figures, must be
  shared under the same licence.
* **Ocean conditions** ([E1], SHOM – IFREMER, WAVEWATCH III hindcast): public,
  **citation mandatory**: Boudière, E., Maisondieu, C., Ardhuin, F., Accensi, M.,
  Pineau-Guillou, L., Lepesqueur, J. (2013). *A suitable metocean hindcast
  database for the design of Marine energy converters.* International Journal
  of Marine Energy 3–4, e40–e52. [doi:10.1016/j.ijome.2013.11.010](https://doi.org/10.1016/j.ijome.2013.11.010)
* **Weather** ([E2], rp5.ua, from Met Office / NOAA data): public; cite the source.
* **European coastline** ([C2], European Environment Agency, "EEA coastline for
  analysis", version 08/07/2013, derived from EU-Hydro and GSHHG): the EEA's
  standard re-use policy allows re-use for commercial or non-commercial
  purposes, free of charge, **provided the source is acknowledged**. Copyright
  holder: Directorate-General for Environment (DG ENV).
  <http://www.eea.europa.eu/legal/copyright>

If you publish work that uses this data, cite Ray et al. (2019) above as well.
The sample in `data/sample/` (24 hours of AIS and weather, and the clipped
coastline) is cut from the same archive, under the same terms ([`data/LICENSE.md`](../data/LICENSE.md)).
