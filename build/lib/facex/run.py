import torch
from .component import run

# define the name of the target attribute e.g. "Gender"
target = "task"
# define the name of the protected attribute e.g. "Gender"
protected = "protected"
# load your model
model = torch.load("your_model.pt")
# define the data directory
data_dir = "<path/to/data>"
# csv should involve all these three columns: img|<task>|<protected>
csv_dir = "<path/to/annotations>.csv"
# set the model's target layer for gradcam
target_layer = "layer4"  # e.g. layer4 from resnet18
# set a specific class ("eg Male") for the target (eg "Gender").
target_class = 1

fig_heatmap, fig_patches, html = run(
    target,
    protected,
    target_class,
    model,
    data_dir,
    csv_dir,
    target_layer,
)

# Save the HTML file
with open("facex_plots.html", "w") as f:
    f.write(html)
