"""Download the US MarineCadastre AIS days and the NOAA shoreline the US dataset needs.

The sources (public data, no login):

* **AIS**: the daily files of the US Coast Guard's AIS broadcasts that
  MarineCadastre.gov / NOAA publish, ``ais-2017-MM-DD.csv.zst``, from
  https://noaaocm.blob.core.windows.net/ais/csv2/csv2017/ (index:
  https://noaaocm.blob.core.windows.net/ais/csv2/csv2017/index.html; catalogue:
  https://hub.marinecadastre.gov).  November and December 2017 are 61 files, about
  13 GB in all (a day is 100 to 250 MB compressed).
* **Shoreline**: the NOAA Medium Resolution Shoreline,
  https://coast.noaa.gov/htdata/Shoreline/us_medium_shoreline.zip (14 MB).

The files go to ``data/UsAIS/csv2017/`` and ``data/UsAIS/shoreline/``, which git
ignores.  Files are fetched one at a time with a progress line.  **A file that is
already in place is never downloaded again** (nothing is deleted or overwritten);
an interrupted download resumes where it stopped (a ``.part`` file and an HTTP
range request), is retried up to five times, and is checked against the size and
the MD5 the server reports.

Usage, from the repository root::

    python scripts/download_us.py                  # 61 days and the shoreline
    python scripts/download_us.py --months 2017-11 # November only
    python scripts/download_us.py --no-shoreline   # AIS only
    python scripts/download_us.py --check          # report what is present, download nothing

The functions (:func:`fetch_days`, :func:`fetch_shoreline`, :func:`report`) are
what notebook 01 calls.  Only the standard library is used.

Terms: MarineCadastre / NOAA data are public (U.S. Government work, no use
restriction; cite MarineCadastre.gov and NOAA, "AIS data for 2017").  The code's
BSD licence does not cover the data.
"""

from __future__ import annotations

import argparse
import base64
import calendar
import hashlib
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "UsAIS"
AIS_DIR = DATA / "csv2017"
SHORELINE_DIR = DATA / "shoreline"
AIS_URL = "https://noaaocm.blob.core.windows.net/ais/csv2/csv2017/"
SHORELINE_URL = "https://coast.noaa.gov/htdata/Shoreline/us_medium_shoreline.zip"
MONTHS = ("2017-11", "2017-12")
HEADERS = {"User-Agent": "catraj-download-us"}


def day_names(months=MONTHS) -> list[str]:
    """``ais-2017-11-01.csv.zst`` ... for every day of ``months`` ("YYYY-MM")."""
    names = []
    for month in months:
        year, mon = (int(p) for p in month.split("-"))
        names += [f"ais-{year:04d}-{mon:02d}-{d:02d}.csv.zst"
                  for d in range(1, calendar.monthrange(year, mon)[1] + 1)]
    return names


def present_days(months=MONTHS, directory: Path = AIS_DIR) -> dict[str, bool]:
    """Which daily files are in ``directory``."""
    return {n: (Path(directory) / n).is_file() for n in day_names(months)}


def shoreline_present(directory: Path = SHORELINE_DIR) -> bool:
    """True when ``directory`` holds a shapefile."""
    directory = Path(directory)
    return directory.is_dir() and any(directory.glob("*.shp"))


def remote_info(url: str) -> tuple[int | None, str | None]:
    """``(size in bytes, hex MD5)`` the server reports for ``url`` (``None`` when absent)."""
    request = urllib.request.Request(url, method="HEAD", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        size = response.headers.get("Content-Length")
        md5 = response.headers.get("Content-MD5")
    return (int(size) if size else None,
            base64.b64decode(md5).hex() if md5 else None)


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, attempts: int = 5, say=print) -> Path:
    """Stream ``url`` to ``target`` (resuming ``target.part``); never overwrites ``target``.

    Returns ``target``.  Checks the size and, when the server reports one, the MD5.
    """
    target = Path(target)
    if target.exists():
        say("  already in place")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    size, md5 = remote_info(url)
    part = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        if size is not None and done >= size:
            break
        request = urllib.request.Request(url, headers=dict(HEADERS))
        if done:
            # a closed range: this server ignores an open-ended one ("bytes=N-")
            request.add_header("Range", f"bytes={done}-{size - 1 if size else ''}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if done and response.status != 206:       # server ignored the range: restart
                    done = 0
                with open(part, "ab" if done else "wb") as f:
                    started, got, shown = time.monotonic(), 0, 0
                    total = f" / {size / 1e6:,.1f}" if size else ""
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if got - shown >= (25 << 20):             # a line every 25 MB
                            shown = got
                            rate = got / max(time.monotonic() - started, 1e-6) / 1e6
                            print(f"  {(done + got) / 1e6:7,.1f}{total} MB  ({rate:4.1f} MB/s)",
                                  flush=True)
                    rate = got / max(time.monotonic() - started, 1e-6) / 1e6
                    print(f"  {(done + got) / 1e6:7,.1f}{total} MB  ({rate:4.1f} MB/s)", flush=True)
            if size is None:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            print(f"\n  interrupted ({error}); attempt {attempt} of {attempts}, resuming ...")
            time.sleep(min(30, 2 ** attempt))
    if not part.exists() or (size is not None and part.stat().st_size != size):
        raise SystemExit(f"could not download {url}; run again to resume")
    if md5 is not None and md5_of(part) != md5:
        part.unlink()
        raise SystemExit(f"checksum mismatch for {target.name}: the partial file was "
                         "deleted, run again to download it afresh")
    part.replace(target)
    return target


def fetch_days(months=MONTHS, directory: Path = AIS_DIR, say=print) -> list[Path]:
    """The daily files of ``months``, one at a time; existing files are skipped."""
    directory = Path(directory)
    names = day_names(months)
    paths = []
    for i, name in enumerate(names, start=1):
        say(f"[{i}/{len(names)}] {name}")
        t = time.perf_counter()
        target = directory / name
        existed = target.exists()
        download(AIS_URL + name, target, say=say)
        if not existed:
            say(f"  {time.perf_counter() - t:.0f} s")
        paths.append(target)
    return paths


def fetch_shoreline(directory: Path = SHORELINE_DIR, say=print) -> Path | None:
    """The NOAA Medium Resolution Shoreline into ``directory``; skipped if a ``.shp`` is there.

    The zip is downloaded (unless present) and extracted, member by member, without
    overwriting; it is kept.  Returns the shapefile.
    """
    directory = Path(directory)
    say(f"NOAA Medium Resolution Shoreline -> {directory}")
    if shoreline_present(directory):
        say("  already in place")
        return sorted(directory.glob("*.shp"))[0]
    zpath = directory / "us_medium_shoreline.zip"
    download(SHORELINE_URL, zpath, say=say)
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            if not (directory / member).exists():
                say(f"  extracting {member}")
                z.extract(member, directory)
    return sorted(directory.glob("*.shp"))[0]


def report(months=MONTHS, ais_dir: Path = AIS_DIR, shoreline_dir: Path = SHORELINE_DIR,
           say=print) -> bool:
    """Print what is present; ``True`` when every day and the shoreline are."""
    days = present_days(months, ais_dir)
    have = sum(days.values())
    size = sum((Path(ais_dir) / n).stat().st_size for n, ok in days.items() if ok)
    say(f"  AIS days: {have}/{len(days)} in {Path(ais_dir)}  ({size / 1e9:.1f} GB)")
    missing = [n for n, ok in days.items() if not ok]
    if missing:
        say(f"  missing: {missing[0]} ... {missing[-1]}" if len(missing) > 2 else f"  missing: {missing}")
    shore = shoreline_present(shoreline_dir)
    say(f"  shoreline: {'present' if shore else 'missing'} in {Path(shoreline_dir)}")
    return have == len(days) and shore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the MarineCadastre AIS days (Nov-Dec 2017) and the NOAA shoreline "
                    "into data/UsAIS/.")
    parser.add_argument("--months", default=",".join(MONTHS),
                        help="comma-separated YYYY-MM (default: 2017-11,2017-12)")
    parser.add_argument("--no-shoreline", action="store_true", help="skip the shoreline")
    parser.add_argument("--no-ais", action="store_true", help="skip the AIS days")
    parser.add_argument("--check", action="store_true",
                        help="only report what is present; download nothing")
    args = parser.parse_args(argv)
    months = tuple(m.strip() for m in args.months.split(",") if m.strip())

    if args.check:
        print(f"data/UsAIS for {', '.join(months)}:")
        return 0 if report(months) else 1
    print("MarineCadastre / NOAA AIS and shoreline -> data/UsAIS/ (existing files are kept)")
    if not args.no_ais:
        fetch_days(months)
    if not args.no_shoreline:
        fetch_shoreline()
    print("\ndone:")
    report(months)
    return 0


if __name__ == "__main__":
    sys.exit(main())
