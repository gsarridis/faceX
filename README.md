# FaceX : Understanding Face Attribute Classifiers through Summary Model Explanations

### This is the official pytorch implementation of FaceX
![FaceX overview](images/facex.JPG)
##### FaceX employs 19 facial regions and accessories to provide explanations (left: face regions, right: hat and glasses). Blue to red colors indicate low to high importance, respectively.  The provided illustration answers the questions "where does a model focus on?" and "what visual features trigger its focus?" through heatmap and high-impact patches visualizations, respectively. This example depicts a biased gender classifier trained on CelebA that effectively uses the Wearing_Lipstick attribute as a shortcut to predict Gender. Note that FaceX is compatible with any face dataset.

### How to run
#### First install the facextool library
```
pip install facextool
```
#### Run faceX
```
from facex.component import run

# define the name of the target attribute e.g. "Gender" 
target = "task"
# define the name of the protected attribute e.g. "Gender" 
protected = "protected"
# define the path of the pretrained model
model_path = "<path/to/model>.pt"
# define the data directory 
data_dir = "<path/to/data>"
# csv should involve all these three columns: img|<task>|<protected>
csv_dir = "<path/to/annotations>.csv"
# set the model's target layer for gradcam
target_layer = "layer4"
# set a specific class ("eg Male") for the target (eg "Gender").
target_class = 1

fig_heatmap, fig_patches, html = run(
    target,
    protected,
    target_class,
    model_path,
    data_dir,
    csv_dir,
    target_layer,
)

# Save the HTML file
with open("facex_plots.html", "w") as f:
    f.write(html)

```

