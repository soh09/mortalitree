"""
Hardcoded bboxes (WGS-84 lon/lat) inside California national/state forests.
Each box is a few km on a side, sited away from towns/highways to keep the
imagery mostly forested. Used by download_naip.py to query NAIP STAC items.
"""

CA_FOREST_BBOXES: list[dict] = [
    {
        "name": "sierra_shaver_lake",
        # Sierra NF, mixed conifer above Shaver Lake
        "bbox": [-119.345, 37.105, -119.245, 37.175],
    },
    {
        "name": "tahoe_north_of_truckee",
        # Tahoe NF, conifer ridges N of Truckee
        "bbox": [-120.260, 39.420, -120.150, 39.500],
    },
    {
        "name": "mendocino_snow_mountain",
        # Mendocino NF / Snow Mountain Wilderness
        "bbox": [-122.840, 39.360, -122.730, 39.440],
    },
    {
        "name": "sequoia_kern_plateau",
        # Sequoia NF, Kern Plateau conifer
        "bbox": [-118.330, 36.090, -118.230, 36.170],
    },
    {
        "name": "redwood_prairie_creek",
        # Redwood NSP, Prairie Creek redwoods
        "bbox": [-124.040, 41.360, -123.960, 41.430],
    },
    {
        "name": "klamath_marble_mountain",
        # Klamath NF, Marble Mountain Wilderness
        "bbox": [-123.260, 41.560, -123.150, 41.640],
    },
    {
        "name": "six_rivers_trinity",
        # Six Rivers NF, Trinity area
        "bbox": [-123.700, 40.880, -123.600, 40.960],
    },
    {
        "name": "lassen_south",
        # Lassen NF, S of the national park
        "bbox": [-121.480, 40.290, -121.380, 40.370],
    },
    {
        "name": "plumas_bucks_lake",
        # Plumas NF, Bucks Lake Wilderness
        "bbox": [-121.270, 39.870, -121.170, 39.940],
    },
    {
        "name": "los_padres_big_sur",
        # Los Padres NF, Big Sur backcountry
        "bbox": [-121.640, 36.190, -121.540, 36.270],
    },
    {
        "name": "czu_big_basin",
        # Big Basin Redwoods SP, core of the 2020 CZU Lightning Complex burn scar
        "bbox": [-122.260, 37.150, -122.160, 37.220],
    },
    # --- 2020 fire burn scars (added for tree-mortality training signal) ---
    # NAIP is 2022, so these capture ~2 years post-fire: standing dead + early
    # recovery, which is the regime the downstream mortality task cares about.
    {
        "name": "creek_huntington_lake",
        # Creek Fire — Sierra NF, Huntington Lake; dense conifer, severe mortality
        "bbox": [-119.270, 37.205, -119.170, 37.275],
    },
    {
        "name": "castle_mountain_home",
        # Castle Fire (SQF Complex) — Mountain Home / Tule River; conifer + giant
        # sequoia groves, catastrophic mortality
        "bbox": [-118.700, 36.190, -118.600, 36.260],
    },
    {
        "name": "north_complex_feather",
        # North Complex (Bear Fire) — Plumas NF, Middle Fork Feather; mixed conifer
        "bbox": [-121.300, 39.720, -121.200, 39.790],
    },
    {
        "name": "scu_henry_coe",
        # SCU Lightning Complex — Diablo Range, Henry W. Coe SP. NOTE: oak woodland
        # / gray pine / chaparral, not conifer — a distinct biome from the others.
        "bbox": [-121.500, 37.150, -121.400, 37.220],
    },
]
