from .component import run

target = "task"
protected = "protected"
target_class = 1
model_path = "/home/gsarridis/projects/gender-bias-xai/logs/a_config_Male_Wearing_Lipstick_0.01/best_ep.pt"
data_dir = "/fssd4/user-data/gsarridis/race_per_7000"
csv_dir = "/fssd4/user-data/gsarridis/bupt_anno.csv"

target_layer = "layer4"

_, _, _ = run(
    target,
    protected,
    target_class,
    model_path,
    data_dir,
    csv_dir,
    target_layer,
)
