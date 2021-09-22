import json
import pandas as pd
import plotly.express as px

def plot_losses(title, name, experiments, normalize=False):
    dataset = []
    for exp, losses in experiments.items():
        if normalize:
            # Normalize as % of x min loss value
            min_loss = min(losses)
            losses = [loss / min_loss for loss in losses]
        dataset.extend([[exp, i, loss] for i, loss in enumerate(losses)])
    df = pd.DataFrame(dataset, columns=["er", "epoch", "loss"])
    fig = px.line(df, x="epoch", y="loss",  hover_name="epoch",
        line_shape="spline", render_mode="svg", title=title)
    # fig.show()
    fig.write_image("data/from_prodigy_to_spacy/output/images/loss_fig1.png")

    
def get_losses(path):
    records = [json.loads(line) for line in open(path)]
    losses = [rec["epoch_loss"] for rec in records]
    return losses

width128= get_losses("data/from_prodigy_to_spacy/output/log.jsonl")

# width256= get_losses("cw256/log.jsonl")
plot_losses("Losses", "Width", {"w128": width128}, normalize=False)