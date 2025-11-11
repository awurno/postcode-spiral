#!/usr/bin/env python3
"""
CLI tool to fetch postcode6 features from PDOK and draw a line connecting centroids.

Supports three modes:
- --bbox xmin,ymin,xmax,ymax  (RD New coordinates, EPSG:28992)
- --municipality NAME         (resolved via Nominatim)
- --province NAME             (resolved via Nominatim)

Output formats:
- .html → interactive folium map (WGS84)
- .png / .jpg / .jpeg → static matplotlib map (RD coordinates)
"""
import argparse
import sys
import re
from typing import Tuple, Optional

import requests
import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from pyproj import Transformer

# Optional plotting libs (only required for static output)
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    import numpy as np
except Exception:
    plt = None
    MplPolygon = None
    np = None


PDOK_PC6_WFS = "https://service.pdok.nl/cbs/postcode6/2024/wfs/v1_0"
PDOK_BESTUUR_WFS = "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0"

# Coordinate transformers
TO_WGS = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    """Parse bbox as xmin,ymin,xmax,ymax (EPSG:28992)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be xmin,ymin,xmax,ymax (EPSG:28992)")
    return tuple(float(p) for p in parts)


def nominatim_bbox_for(name: str) -> Tuple[float, float, float, float]:
    """Resolve a place name to RD bbox using Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": name, "format": "json", "limit": 1, "countrycodes": "nl"}
    headers = {"User-Agent": "postcode-map-tool/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError(f"Could not resolve name: {name}")
    # bbox: south, north, west, east (WGS84)
    south, north, west, east = map(float, data[0]["boundingbox"])
    # transform to RD
    xmin, ymin = TO_RD.transform(west, south)
    xmax, ymax = TO_RD.transform(east, north)
    return (xmin, ymin, xmax, ymax)


def fetch_boundary_geometry(name: str, boundary_type: str = "gemeenten") -> Optional[dict]:
    from io import BytesIO

    if boundary_type != 'gemeenten':
        print('boundary geometry only works for gemeenten for now.')
        return None

    url = "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <wfs:GetFeature service="WFS" version="2.0.0"
        xmlns:wfs="http://www.opengis.net/wfs/2.0"
        xmlns:fes="http://www.opengis.net/fes/2.0">
    <wfs:Query typeNames="Gemeentegebied">
        <fes:Filter>
        <fes:PropertyIsEqualTo>
            <fes:ValueReference>naam</fes:ValueReference>
            <fes:Literal>{name}</fes:Literal>
        </fes:PropertyIsEqualTo>
        </fes:Filter>
    </wfs:Query>
    </wfs:GetFeature>
    """
    headers = {"Content-Type": "text/xml; charset=UTF-8"}
    r = requests.post(url, data=xml.encode("utf-8"), headers=headers, params={"outputFormat": "application/json"})
    print(r.url)
    r.raise_for_status()

    gdf = gpd.read_file(BytesIO(r.content))
    print(gdf.columns)
    print(gdf['naam'])

    return gdf


def _fetch_boundary_geometry(name: str, boundary_type: str = "gemeenten") -> Optional[dict]:
    """Fetch an administrative boundary polygon from PDOK bestuurlijkegebieden WFS.
    
    boundary_type: 'gemeenten' for municipalities, 'provincies' for provinces.
    Returns a shapely geometry or None if not found.
    """
    # Try a safe approach: fetch a reasonable number of features and match the
    # provided name case-insensitively against string properties. This avoids
    # relying on server-side CQL dialects and is robust to property naming.
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": boundary_type,
        "outputFormat": "application/json",
        "srsName": "EPSG:28992",
        "count": 500,
    }
    try:
        r = requests.get(PDOK_BESTUUR_WFS, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        lname = name.strip().lower()
        for feat in features:
            props = feat.get("properties", {}) or {}
            for v in props.values():
                if isinstance(v, str) and lname == v.strip().lower():
                    geom_dict = feat.get("geometry")
                    if geom_dict:
                        return shape(geom_dict)
        # if exact match not found, try substring match
        for feat in features:
            props = feat.get("properties", {}) or {}
            for v in props.values():
                if isinstance(v, str) and lname in v.strip().lower():
                    geom_dict = feat.get("geometry")
                    if geom_dict:
                        return shape(geom_dict)
    except Exception as e:
        print(f"Warning: Could not fetch boundary for '{name}': {e}")
    return None


def fetch_postcodes_geojson(bbox: Tuple[float, float, float, float] = None, geom = None) -> dict:
    """Fetch postcode6 GeoJSON from PDOK WFS.

    If ``geom`` (a shapely geometry in RD coords) is provided we first request
    features within the geometry's bbox and then filter them locally by
    intersection with the provided geometry. This avoids relying on CQL
    dialects on the server.
    """
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": "postcode6",
        "outputFormat": "application/json",
        "srsName": "EPSG:28992",
    }

    if geom is not None:
        bounds = geom.bounds
        params["bbox"] = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]},EPSG:28992"
    elif bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        params["bbox"] = f"{xmin},{ymin},{xmax},{ymax},EPSG:28992"
    else:
        raise ValueError("Either bbox or geom must be provided")

    r = requests.get(PDOK_PC6_WFS, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    if geom is not None:
        # filter client-side by intersection with the provided geometry
        feats = []
        for f in data.get("features", []):
            geom_dict = f.get("geometry")
            if not geom_dict:
                continue
            try:
                fshape = shape(geom_dict)
            except Exception:
                continue
            if fshape.intersects(geom):
                feats.append(f)
        return {"type": "FeatureCollection", "features": feats}

    return data


def _extract_postcode(props: dict) -> Optional[str]:
    """Extract pc6 (1234AB) or pc4 (1234) from properties; return normalized string or None."""
    six_re = re.compile(r"\b(\d{4}[A-Za-z]{2})\b")
    four_re = re.compile(r"\b(\d{4})\b")

    # scan all properties
    for v in props.values():
        if isinstance(v, str):
            m6 = six_re.search(v)
            if m6:
                return m6.group(1).upper()
            m4 = four_re.search(v)
            if m4:
                return m4.group(1)
        elif isinstance(v, (int, float)):
            if 1000 <= int(v) <= 9999:
                return f"{int(v):04d}"

    # fallback: check common property names
    for key in ("pc6", "postcode6", "postcode", "pc4", "postcode4"):
        if key in props:
            v = props[key]
            if isinstance(v, str):
                m6 = six_re.search(v)
                if m6:
                    return m6.group(1).upper()
            if isinstance(v, (int, float)) and 1000 <= int(v) <= 9999:
                return f"{int(v):04d}"

    return None


def geojson_to_gdf(geojson: dict) -> gpd.GeoDataFrame:
    """Convert postcode6 GeoJSON to GeoDataFrame with postcode, centroid columns."""
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:28992")

    # extract postcode from original properties (from_features unpacks them into columns)
    # so we need to re-extract from the raw features
    features = geojson.get("features", [])
    postcode_raws = [
        _extract_postcode(f.get("properties", {})) for f in features
    ]
    gdf["postcode_raw"] = postcode_raws

    # parse postcode into numeric part and suffix
    def parse_pc(pc_str):
        if not pc_str:
            return None, None
        if len(pc_str) == 6 and pc_str[:4].isdigit() and pc_str[4:].isalpha():
            return int(pc_str[:4]), pc_str[4:].upper()
        if len(pc_str) == 4 and pc_str.isdigit():
            return int(pc_str), None
        return None, None

    gdf[["postcode", "pc6_suffix"]] = gdf["postcode_raw"].apply(
        lambda x: pd.Series(parse_pc(x))
    )

    # compute RD centroids
    gdf["centroid_rd"] = gdf.geometry.centroid

    # transform to WGS84 for folium (compute centroid in projected CRS first, then transform)
    gdf_wgs = gdf.to_crs("EPSG:4326")
    # centroid_wgs should be derived from centroid_rd (already computed in projected space)
    gdf["centroid_wgs"] = gdf["centroid_rd"].apply(
        lambda pt: TO_WGS.transform(pt.x, pt.y) if not pt.is_empty else None
    )
    # Convert tuples to Point objects for consistency
    from shapely.geometry import Point as ShapelyPoint
    gdf["centroid_wgs"] = gdf["centroid_wgs"].apply(
        lambda coord: ShapelyPoint(coord) if coord else None
    )

    return gdf[["postcode_raw", "postcode", "pc6_suffix", "centroid_rd", "centroid_wgs", "geometry"]]


def sort_key(row: pd.Series, precision: str = "4") -> tuple:
    """Compute sort key for a row depending on precision."""
    pc = row["postcode"]
    suffix = row["pc6_suffix"] or ""

    if pd.isna(pc):
        # fallback: sort by RD centroid
        c = row["centroid_rd"]
        return (1, c.x, c.y)

    if precision == "6":
        return (0, pc, suffix.upper())
    if precision == "5":
        return (0, pc, suffix[0].upper() if suffix else "")
    # precision 4
    return (0, pc)


def build_map(gdf: gpd.GeoDataFrame, out_path: str, precision: str = "4", show_labels: bool = True):
    """Build interactive or static map depending on output extension."""
    out_lower = out_path.lower()
    is_static = out_lower.endswith((".png", ".jpg", ".jpeg"))

    # sort by precision
    gdf = gdf.assign(sort_key=gdf.apply(lambda r: sort_key(r, precision), axis=1)).sort_values("sort_key")

    if is_static:
        _build_static_map(gdf, out_path, show_labels)
    else:
        _build_folium_map(gdf, out_path, show_labels)


def _build_static_map(gdf: gpd.GeoDataFrame, out_path: str, show_labels: bool):
    """Build a static matplotlib map in RD coordinates."""
    if plt is None or np is None or MplPolygon is None:
        raise RuntimeError("matplotlib and numpy required for static output. Install requirements.txt")

    fig, ax = plt.subplots(figsize=(10, 10))

    # draw polygons
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        for g in geoms:
            try:
                coords = np.array(g.exterior.coords)
                patch = MplPolygon(coords, closed=True, fill=False, edgecolor="#444", linewidth=0.7)
                ax.add_patch(patch)
            except Exception:
                continue

    # plot centroids and line
    centroids = [c for c in gdf["centroid_rd"] if not c.is_empty]
    xs = [c.x for c in centroids]
    ys = [c.y for c in centroids]

    if xs and ys:
        ax.scatter(xs, ys, s=1, c="red")
        ax.plot(xs, ys, color="blue", linewidth=1)

        if show_labels:
            for x, y, raw in zip(xs, ys, gdf["postcode_raw"]):
                if raw:
                    ax.text(x + 10, y + 10, raw, fontsize=8, color="#003366",
                           bbox={"facecolor": "white", "alpha": 0.7, "pad": 1})

    ax.set_aspect("equal", adjustable="datalim")
    if xs and ys:
        dx = max(1.0, (max(xs) - min(xs)) * 0.05)
        dy = max(1.0, (max(ys) - min(ys)) * 0.05)
        ax.set_xlim(min(xs) - dx, max(xs) + dx)
        ax.set_ylim(min(ys) - dy, max(ys) + dy)

    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote static map to {out_path}")


def _build_folium_map(gdf: gpd.GeoDataFrame, out_path: str, show_labels: bool):
    """Build an interactive folium map in WGS84."""
    # compute mean center in WGS84
    gdf_wgs = gdf.to_crs("EPSG:4326")
    bounds = gdf_wgs.total_bounds
    center_y = (bounds[1] + bounds[3]) / 2
    center_x = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[center_y, center_x], zoom_start=12)

    # add polygon outlines and markers
    for _, row in gdf.iterrows():
        raw = row["postcode_raw"]
        label = raw or ""

        # polygon outline (convert back to WGS84)
        geom_wgs = gpd.GeoSeries([row.geometry], crs="EPSG:28992").to_crs("EPSG:4326")[0]
        try:
            folium.GeoJson(
                geom_wgs.__geo_interface__,
                style_function=lambda feat: {"color": "#444", "weight": 1, "fill": False},
            ).add_to(m)
        except Exception:
            pass

        # marker and label
        centroid_wgs = row["centroid_wgs"]
        if not centroid_wgs.is_empty:
            folium.CircleMarker(
                [centroid_wgs.y, centroid_wgs.x], radius=3, color="red", popup=label
            ).add_to(m)

            if label and show_labels:
                try:
                    folium.Marker(
                        [centroid_wgs.y, centroid_wgs.x],
                        icon=folium.DivIcon(
                            html=f"<div style='font-size:10px;color:#003366;background:rgba(255,255,255,0.7);padding:1px 3px;border-radius:2px'>{label}</div>"
                        ),
                    ).add_to(m)
                except Exception:
                    pass

    # connecting line (WGS84)
    centroids_wgs = [c for c in gdf_wgs["centroid_wgs"] if not c.is_empty]
    if centroids_wgs:
        line = [[c.y, c.x] for c in centroids_wgs]
        folium.PolyLine(line, color="blue", weight=3).add_to(m)

    m.save(out_path)
    print(f"Wrote map to {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Create postcode line maps using PDOK postcode6 data (RD coordinates)."
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--bbox", type=parse_bbox, help="xmin,ymin,xmax,ymax (EPSG:28992)")
    grp.add_argument("--municipality", help="Municipality name (NL)")
    grp.add_argument("--province", help="Province name (NL)")
    p.add_argument(
        "--precision",
        choices=("4", "5", "6"),
        default="4",
        help="Precision: 4 (numeric), 5 (+ first letter), 6 (+ full suffix). Default: 4",
    )
    p.add_argument("--no-labels", dest="labels", action="store_false", default=True, help="Hide postcode labels on the map")
    p.add_argument("--output", default="output.html", help="Output file (.html, .png, etc)")
    args = p.parse_args(argv)

    try:
        bbox = None
        geom = None
        
        if args.bbox:
            bbox = args.bbox
            print(f"Using bbox: {bbox}")
        elif args.municipality:
            geom = fetch_boundary_geometry(args.municipality, "gemeenten")
            
            if geom is not None:
                if not geom.empty:
                    print(f"Using municipality boundary: {args.municipality}")
            else:
                # Fallback to bbox if boundary lookup fails
                bbox = nominatim_bbox_for(args.municipality)
                print(f"Fallback to bbox: {bbox}")
        elif args.province:
            print('Province not supported for now.')
            exit
            geom = fetch_boundary_geometry(args.province, "provincies")
            if geom:
                print(f"Using province boundary: {args.province}")
            else:
                # Fallback to bbox if boundary lookup fails
                bbox = nominatim_bbox_for(args.province)
                print(f"Fallback to bbox: {bbox}")

        gj = fetch_postcodes_geojson(bbox=bbox, geom=geom)
        gdf = geojson_to_gdf(gj)

        if gdf.empty:
            print("No postcode features found in bbox. Try expanding the area.")
            sys.exit(2)

        build_map(gdf, args.output, precision=args.precision, show_labels=args.labels)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
