"""Download the public data the tutorials, workflows and benchmarks use.

Everything comes from one Zenodo record:

    Ray, C., Dréo, R., Camossi, E., Jousselme, A.-L., Iphar, C. (2019).
    Heterogeneous integrated dataset for Maritime Intelligence, Surveillance,
    and Reconnaissance. Data in Brief 25, 104141. doi:10.1016/j.dib.2019.104141
    Data: doi:10.5281/zenodo.1167595 -- https://zenodo.org/records/1167595

The files go to ``data/zenodo_1167595/``, which git ignores. Each archive is
checked against the MD5 checksum published on the record, then only the files
the code reads are extracted. An interrupted download resumes where it stopped,
and anything already in place is skipped, so running the script again is safe.

Usage, from the repository root, in the ``catraj`` environment::

    python scripts/download_data.py                # everything (~773 MB download)
    python scripts/download_data.py annotation     # tutorial 1 only (~203 MB)
    python scripts/download_data.py cadp           # tutorial 2 and the benchmarks (~771 MB)
    python scripts/download_data.py --check        # report what is present, download nothing

Options::

    --months feb,jan    sea-state months to extract (oct nov dec jan feb mar, or all);
                        the tutorials need only feb
    --remove-zips       delete each archive once extracted (saves ~770 MB; a later
                        run, or DOWNLOAD = True in a notebook, downloads it again)

Only the Python standard library is used, so the script runs before anything
else is installed. The full guide is ``docs/getting_the_data.md``.

Licences: the AIS data is CC BY-NC-SA 4.0 (attribute NARI, Brest; no
commercial use; derived data under the same licence). Sea state (SHOM-IFREMER,
WAVEWATCH III) is public with citation mandatory: Boudière et al. (2013),
doi:10.1016/j.ijome.2013.11.010. Weather (rp5.ua, from Met Office / NOAA data)
is public; cite the source.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo_1167595"
RECORD = "https://zenodo.org/records/1167595/files/"

MONTHS = {"oct": "october", "nov": "november", "dec": "december",
          "jan": "january", "feb": "february", "mar": "march"}

# Zenodo file name -> MD5 published on the record, size in bytes, the folder it
# is extracted to, and the files extracted from it (None: every file; tiny archives).
ARCHIVES = {
    "[P1] AIS Data.zip": dict(
        md5="90bec666a3c3474e18bb57676348b655", size=201_433_686, folder="P1_AIS_Data",
        members=["nari_dynamic.csv", "nari_static.csv", "README_parameters.txt"]),
    "[P1] AIS Status, Codes and Types.zip": dict(
        md5="6f9bb328d6e657b16fea55210a6e56cf", size=7_934, folder="P1_AIS_Codes",
        members=None),
    "[E2] Weather Conditions.zip": dict(
        md5="d8ac0644a670f584daaa4a450cf2baa7", size=1_346_914, folder="E2_Weather_Conditions",
        members=None),
    "[E1] Ocean Conditions.zip": dict(
        md5="20a1c8f8a6bbbd9bdf9821f938f22add", size=569_784_052, folder="E1_ocean",
        members=["README_parameters.txt"]),          # + oc_<month>.csv, from --months
}

# What each part of the repository reads.
TARGETS = {
    "annotation": ["[P1] AIS Data.zip", "[P1] AIS Status, Codes and Types.zip",
                   "[E2] Weather Conditions.zip"],
    "cadp": ["[P1] AIS Data.zip", "[E1] Ocean Conditions.zip"],
}
TARGETS["all"] = list(dict.fromkeys(TARGETS["annotation"] + TARGETS["cadp"]))


def zip_path(name: str) -> Path:
    """Local file name of an archive; the same name the notebooks use, so they find it."""
    return DATA / (name.replace("[", "").replace("] ", "_").replace(",", "").replace(" ", "_"))


def members_of(name: str, months: list[str]) -> list[str] | None:
    members = ARCHIVES[name]["members"]
    if name == "[E1] Ocean Conditions.zip":
        members = members + [f"oc_{MONTHS[m]}.csv" for m in months]
    return members


def present(name: str, months: list[str]) -> bool:
    """True when every file needed from this archive is already extracted."""
    members = members_of(name, months)
    folder = DATA / ARCHIVES[name]["folder"]
    if members is None:
        return folder.is_dir() and any(folder.iterdir())
    return all((folder / m).exists() for m in members)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, target: Path, size: int, attempts: int = 5) -> None:
    """Stream one file from Zenodo to ``target``, resuming a partial download."""
    url = RECORD + urllib.parse.quote(name) + "?download=1"
    part = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        if done >= size:
            break
        request = urllib.request.Request(url, headers={"User-Agent": "catraj-download-data"})
        if done:
            request.add_header("Range", f"bytes={done}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if done and response.status != 206:      # server ignored the range: restart
                    done = 0
                with open(part, "ab" if done else "wb") as f:
                    started, got = time.monotonic(), 0
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        rate = got / max(time.monotonic() - started, 1e-6) / 1e6
                        print(f"\r  {(done + got) / 1e6:7,.1f} / {size / 1e6:,.1f} MB"
                              f"  ({rate:4.1f} MB/s)", end="", flush=True)
            print()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            print(f"\n  interrupted ({error}); attempt {attempt} of {attempts}, resuming ...")
            time.sleep(min(30, 2 ** attempt))
    if not part.exists() or part.stat().st_size != size:
        raise SystemExit(f"could not download {name}; run the script again to resume, "
                         f"or download it by hand (see docs/getting_the_data.md)")
    part.replace(target)


def fetch(name: str, months: list[str], remove_zip: bool) -> None:
    spec = ARCHIVES[name]
    folder = DATA / spec["folder"]
    print(f"\n{name}  ({spec['size'] / 1e6:,.1f} MB)  ->  {folder.relative_to(ROOT).as_posix()}/")
    if present(name, months):
        print("  already in place")
        return
    zpath = zip_path(name)
    if not zpath.exists():
        download(name, zpath, spec["size"])
    print("  checking MD5 ...", end=" ", flush=True)
    if md5(zpath) != spec["md5"]:
        zpath.unlink()
        raise SystemExit(f"checksum mismatch for {zpath.name}: the file was deleted, "
                         f"run the script again to download it afresh")
    print("ok")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        wanted = members_of(name, months)
        for member in z.namelist() if wanted is None else wanted:
            if not (folder / member).exists():
                print(f"  extracting {member}")
                z.extract(member, folder)
    if remove_zip:
        zpath.unlink()
        print(f"  removed {zpath.name}")


def report(names: list[str], months: list[str]) -> bool:
    ok = True
    for name in names:
        here = present(name, months)
        ok &= here
        folder = (DATA / ARCHIVES[name]["folder"]).relative_to(ROOT).as_posix()
        print(f"  [{'x' if here else ' '}] {name:<40} {folder}/")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the Zenodo data (doi:10.5281/zenodo.1167595) into data/zenodo_1167595/.",
        epilog="Full guide: docs/getting_the_data.md")
    parser.add_argument("target", nargs="?", default="all", choices=sorted(TARGETS),
                        help="annotation: tutorial 1; cadp: tutorial 2 and the benchmarks; "
                             "all (default): both")
    parser.add_argument("--months", default="feb",
                        help="sea-state months to extract: comma-separated from "
                             "oct,nov,dec,jan,feb,mar, or 'all' (default: feb)")
    parser.add_argument("--remove-zips", action="store_true",
                        help="delete each archive after extracting it")
    parser.add_argument("--check", action="store_true",
                        help="only report what is present; download nothing")
    args = parser.parse_args(argv)

    months = list(MONTHS) if args.months == "all" else [m.strip()[:3].lower()
                                                        for m in args.months.split(",")]
    if unknown := [m for m in months if m not in MONTHS]:
        parser.error(f"unknown month(s) {unknown}; use {', '.join(MONTHS)} or all")
    names = TARGETS[args.target]

    if args.check:
        print(f"data for '{args.target}' in {DATA.relative_to(ROOT).as_posix()}/:")
        return 0 if report(names, months) else 1

    total = sum(ARCHIVES[n]["size"] for n in names
                if not present(n, months) and not zip_path(n).exists())
    print(f"Zenodo record 10.5281/zenodo.1167595 -> {DATA.relative_to(ROOT).as_posix()}/")
    print(f"to download: {total / 1e6:,.0f} MB")
    print("The AIS data is CC BY-NC-SA 4.0: attribute NARI, no commercial use, "
          "share derived data alike.")
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        fetch(name, months, args.remove_zips)

    print("\ndone:")
    report(names, months)
    return 0


if __name__ == "__main__":
    sys.exit(main())
