import modal

VOLUME_NAME = 'mot' # change this

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements('../requirements.txt')  # relative path now
)

app = modal.App("mortalitree", image=image)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
DATA_DIR = "/data"

@app.function(volumes={DATA_DIR: volume}, timeout=10_000)
def download_treeboxes():
    from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
    ds = TreeBoxesDataset(root_dir=DATA_DIR, download=True, include_unsupervised=True)
    volume.commit()
    return {"n_images": len(ds), "data_dir": DATA_DIR}

@app.function(volumes={DATA_DIR: volume})
def list_data():
    import os
    for root, dirs, files in os.walk(DATA_DIR):
        depth = root.replace(DATA_DIR, "").count(os.sep)
        if depth > 2:
            continue
        print(root, "->", len(files), "files")

@app.local_entrypoint()
def main():
    result = download_treeboxes.remote()
    print(result)
    list_data.remote()