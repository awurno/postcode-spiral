"""Unit tests for postcode parsing and sorting logic."""
import json
import geopandas as gpd
from shapely.geometry import Point, Polygon

from src import map_postcodes as mp


def test_extract_postcode_pc6():
    """Test extraction of pc6 (1234AB) from properties."""
    props = {"pc6": "1234AB"}
    result = mp._extract_postcode(props)
    assert result == "1234AB"


def test_extract_postcode_pc4():
    """Test extraction of pc4 (1234) from properties."""
    props = {"postcode": "1234"}
    result = mp._extract_postcode(props)
    assert result == "1234"


def test_extract_postcode_none():
    """Test extraction when no postcode property exists."""
    props = {"other": "value"}
    result = mp._extract_postcode(props)
    assert result is None


def test_sort_key_precision4():
    """Test sort key for precision 4 (numeric only)."""
    row_dict = {
        "postcode": 1234,
        "pc6_suffix": "AB",
        "centroid_rd": Point(0, 0),
    }
    # Convert to pandas Series
    import pandas as pd
    row = pd.Series(row_dict)
    key = mp.sort_key(row, "4")
    assert key == (0, 1234)


def test_sort_key_precision5():
    """Test sort key for precision 5 (numeric + first letter)."""
    import pandas as pd
    row1 = pd.Series({"postcode": 1234, "pc6_suffix": "AB", "centroid_rd": Point(0, 0)})
    row2 = pd.Series({"postcode": 1234, "pc6_suffix": "AA", "centroid_rd": Point(0, 0)})
    
    key1 = mp.sort_key(row1, "5")
    key2 = mp.sort_key(row2, "5")
    
    assert key1 == (0, 1234, "A")
    assert key2 == (0, 1234, "A")
    # Both should have same key since first letter is 'A'


def test_sort_key_precision6():
    """Test sort key for precision 6 (numeric + full suffix)."""
    import pandas as pd
    row1 = pd.Series({"postcode": 1234, "pc6_suffix": "AB", "centroid_rd": Point(0, 0)})
    row2 = pd.Series({"postcode": 1234, "pc6_suffix": "AA", "centroid_rd": Point(0, 0)})
    
    key1 = mp.sort_key(row1, "6")
    key2 = mp.sort_key(row2, "6")
    
    assert key1 == (0, 1234, "AB")
    assert key2 == (0, 1234, "AA")


def test_geojson_to_gdf_simple():
    """Test conversion of simple GeoJSON to GeoDataFrame."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
                "properties": {"pc6": "1234AB"},
            }
        ],
    }
    
    gdf = mp.geojson_to_gdf(geojson)
    
    assert len(gdf) == 1
    assert gdf.iloc[0]["postcode_raw"] == "1234AB"
    assert gdf.iloc[0]["postcode"] == 1234
    assert gdf.iloc[0]["pc6_suffix"] == "AB"
