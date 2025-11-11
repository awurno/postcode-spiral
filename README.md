# postcode_map — postcode line map tool

This small tool fetches Dutch `postcode4` features from PDOK and draws a line connecting centroids inside a requested area.

Quick start (Windows PowerShell):

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the CLI by bbox in RD New coordinates (EPSG:28992) as: xmin,ymin,xmax,ymax. Example values below are placeholders — use your RD bbox.

```powershell
python src/map_postcodes.py --bbox 121000,487000,122000,488000 --output area.html
```

Or by municipality/province name (resolved via Nominatim and transformed to RD internally):

```powershell
python src/map_postcodes.py --municipality "Amsterdam" --output amsterdam.html
```

To request postcode6 (pc6) features and sorting, use `--precision 6`. Example:

```powershell
python src/map_postcodes.py --municipality "Amsterdam" --precision 6 --output ams_pc6.html
```

Precision 5 (pc5) — sorting by first letter of the pc6 suffix

```powershell
# use the postcode6 layer but sort by numeric part then the first letter of the suffix
python src/map_postcodes.py --municipality "Amsterdam" --precision 5 --output ams_pc5.html
```

Toggle persistent on-map labels (default is on):

```powershell
# disable persistent labels and use popups only
python src/map_postcodes.py --municipality "Amsterdam" --no-labels --output ams_nolabels.html
```

- Name resolution uses OpenStreetMap Nominatim (public service); heavy/batch usage may be rate-limited.
- The PDOK WFS endpoint is used with bbox filtering. If no features are returned, expand the bbox or use a different area.
- The connecting line is a simple lon/lat sort of centroids (visualization only, not a routing solution).

Notes & limitations
- Name resolution uses OpenStreetMap Nominatim (public service); heavy/batch usage may be rate-limited.
- The PDOK WFS endpoint is used with bbox filtering and requested in EPSG:28992 (RD New). If no features are returned, expand the bbox or use a different area.
- The connecting line is a simple lon/lat sort of centroids (visualization only, not a routing solution).
- Name resolution uses OpenStreetMap Nominatim (public service); heavy/batch usage may be rate-limited.
- The PDOK WFS endpoint is used with bbox filtering. If no features are returned, expand the bbox or use a different area.
- The connecting line is a simple lon/lat sort of centroids (visualization only, not a routing solution).

License: none specified — please add a LICENSE if you intend to publish.
