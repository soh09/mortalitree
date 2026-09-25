"""Print the anchor config the prebuilt DeepForest model actually uses."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("deepforest==2.1.0", "rasterio", "geopandas", "shapely", "pandas", "numpy", "pillow")
    .env({"PYTHONUNBUFFERED": "1"})
)
app = modal.App("deepforest-inspect", image=image)


@app.function(cpu=2.0, timeout=900)
def inspect():
    from deepforest import main as df_main
    m = df_main.deepforest()
    m.load_model()
    net = m.model
    ag = net.anchor_generator
    print("anchor_generator:", type(ag).__name__)
    print("sizes:        ", ag.sizes)
    print("aspect_ratios:", ag.aspect_ratios)
    print("anchors/location:", ag.num_anchors_per_location())
    head = net.head.classification_head
    print("cls head num_anchors:", getattr(head, "num_anchors", "n/a"))
    print("nms_thresh:", net.nms_thresh, " score_thresh:", net.score_thresh)
    print("detections_per_img:", net.detections_per_img, " topk:", net.topk_candidates)
    print("transform min/max size:", net.transform.min_size, net.transform.max_size)
    return {"sizes": str(ag.sizes), "npl": ag.num_anchors_per_location()}


@app.local_entrypoint()
def main():
    print(inspect.remote())
