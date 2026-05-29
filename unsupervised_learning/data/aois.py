"""
Hardcoded bboxes (WGS-84 lon/lat) inside California national/state forests.
Each box is a few km on a side, sited away from towns/highways to keep the
imagery mostly forested. Used by download_naip.py to query NAIP STAC items.

Copied verbatim from synthetic_nir_dev/NAIP/aois.py so the pretraining pool
draws from the same forest regions as the supervised pipeline.
"""

CA_FOREST_BBOXES: list[dict] = [
    {
        "name": "sierra_shaver_lake",
        "bbox": [-119.345, 37.105, -119.245, 37.175],
    },
    {
        "name": "tahoe_north_of_truckee",
        "bbox": [-120.260, 39.420, -120.150, 39.500],
    },
    {
        "name": "mendocino_snow_mountain",
        "bbox": [-122.840, 39.360, -122.730, 39.440],
    },
    {
        "name": "sequoia_kern_plateau",
        "bbox": [-118.330, 36.090, -118.230, 36.170],
    },
    {
        "name": "redwood_prairie_creek",
        "bbox": [-124.040, 41.360, -123.960, 41.430],
    },
    {
        "name": "klamath_marble_mountain",
        "bbox": [-123.260, 41.560, -123.150, 41.640],
    },
    {
        "name": "six_rivers_trinity",
        "bbox": [-123.700, 40.880, -123.600, 40.960],
    },
    {
        "name": "lassen_south",
        "bbox": [-121.480, 40.290, -121.380, 40.370],
    },
    {
        "name": "plumas_bucks_lake",
        "bbox": [-121.270, 39.870, -121.170, 39.940],
    },
    {
        "name": "los_padres_big_sur",
        "bbox": [-121.640, 36.190, -121.540, 36.270],
    },
    # ---------------- 2020 fire perimeters ----------------
    # Bboxes are ~10 km x 8 km sited in the heart of each burn scar, not the
    # entire perimeter (perimeters are huge — full bboxes would pull dozens
    # of scenes per fire and inflate the data pool out of proportion).
    # Pairing these AOIs with NAIP_YEARS=2020,2022 captures both pre-fire
    # (2020 imagery flown before ignition) and post-fire (2022) spectral
    # signatures, which is exactly the diversity the encoder needs.
    {
        "name": "fire_czu_lightning_complex",
        # CZU Lightning Complex (Aug 2020) — Big Basin Redwoods / Boulder Creek
        "bbox": [-122.270, 37.140, -122.170, 37.220],
    },
    {
        "name": "fire_scu_lightning_complex",
        # SCU Lightning Complex (Aug 2020) — Diablo Range / Henry Coe area
        "bbox": [-121.500, 37.210, -121.400, 37.290],
    },
    {
        "name": "fire_creek",
        # Creek Fire (Sep 2020) — Sierra NF, Huntington Lake / Big Creek
        "bbox": [-119.250, 37.230, -119.150, 37.310],
    },
    {
        "name": "fire_north_complex",
        # North Complex / Bear Fire (Aug 2020) — Plumas NF, Berry Creek
        "bbox": [-121.450, 39.610, -121.350, 39.690],
    },
    {
        "name": "fire_castle_sqf_complex",
        # Castle Fire / SQF Complex (Aug 2020) — Sequoia NF, Giant Sequoia NM
        "bbox": [-118.550, 36.160, -118.450, 36.240],
    },
]
