# Reference Dataset Selection

## Selection

The correctness dataset is **Columbus neighborhood crime data (1980)**, distributed in `libpysal.examples` and drawn from Anselin (1988), *Spatial Econometrics*, Table 12.1.

- Access: installed, stable call `libpysal.examples.get_path("columbus.shp")`.
- License: the files are distributed with the BSD-3-Clause-licensed `libpysal` package; the bundled dataset README states its source but does not provide a separate data-license declaration. This ambiguity is a limitation and the data are never redistributed by this project.
- Observations: 49.
- Geometry: Polygon.
- CRS: absent in the bundled shapefile; topology-based Queen validation is unaffected, but metric distance calculations must not treat these coordinates as a known real-world unit.
- Variables: `CRIME` is primary; `HOVAL` and `INC` support multi-variable validation.
- Reason: it is installed with PySAL, includes stable polygon geometry and meaningful numeric variables, is a canonical Moran example, and needs no download.
- Limitations: too small for credible GPU timing, no declared CRS, and no separate license file. Synthetic datasets are used only for scaling benchmarks, never correctness substitution.

```python
import geopandas as gpd
import libpysal

path = libpysal.examples.get_path("columbus.shp")
gdf = gpd.read_file(path)
y = gdf["CRIME"].to_numpy()
```

Programmatic catalog inspection used `libpysal.examples.available()` and confirmed the dataset is locally installed.

