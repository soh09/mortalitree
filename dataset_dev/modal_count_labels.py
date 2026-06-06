"""Count tiles and annotations across all Weinstein 2020 CSVs on the mot volume."""
from __future__ import annotations
import modal

VOLUME_NAME = "mot"
APP_NAME = "momrtalitree"

image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "rich")
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def count_labels() -> list[dict]:
    from pathlib import Path
    import pandas as pd

    results = []
    label_dir = Path("/data/labels")
    csvs = sorted(label_dir.glob("*.csv"))
    for csv in csvs:
        # skip per-tile chunk files (those live in subdirs, not here)
        try:
            df = pd.read_csv(csv, usecols=["geo_index"])
            n_tiles = df["geo_index"].nunique()
            n_rows = len(df)
            results.append({
                "file": csv.name,
                "tiles": n_tiles,
                "annotations": n_rows,
            })
        except Exception as e:
            results.append({"file": csv.name, "tiles": -1, "annotations": -1, "error": str(e)})
    return results


@app.local_entrypoint()
def main():
    from rich.console import Console
    from rich.table import Table

    console = Console()
    rows = count_labels.remote()

    tbl = Table(title="Weinstein 2020 — tiles & annotations per site CSV")
    tbl.add_column("File", style="cyan")
    tbl.add_column("Tiles (geo_index)", justify="right")
    tbl.add_column("Annotations", justify="right")

    total_tiles = 0
    total_ann = 0
    for r in rows:
        if r.get("error"):
            tbl.add_row(r["file"], "[red]ERR[/red]", r["error"])
        else:
            tbl.add_row(r["file"], f"{r['tiles']:,}", f"{r['annotations']:,}")
            total_tiles += r["tiles"]
            total_ann += r["annotations"]

    tbl.add_section()
    tbl.add_row("[bold]TOTAL", f"[bold]{total_tiles:,}", f"[bold]{total_ann:,}")
    console.print(tbl)
    console.print(f"\n[green]{len(rows)} site CSVs found on volume[/green]")
