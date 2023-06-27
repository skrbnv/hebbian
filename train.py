import torch
import torch.nn as nn
import argparse
import wandb
from tqdm import tqdm
from libs.dataset import generate
from libs.evaluate import test_loop
from libs.trainer import Trainer

# from libs.visualize import (
#    visualize_receptive_field,
#    visualize_filters,
#    visualize_feature_maps,
# )
import libs.models as models
import libs.strategies as strategies
import libs.utils as utils

# from libs.simplenorm import SimpleNorm
from statistics import mean


def test_loops(train_loader, test_loader, model):
    print("├── Training acc")
    test_loop(train_loader, model)
    print("├── Testing acc")
    test_loop(test_loader, model)


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False, help="sync with W&B")
args = parser.parse_args()
WANDB = args.wandb
CONFIG = utils.load_yaml()
if WANDB:
    wprj = wandb.init(
        project=CONFIG.wandb.project,
        name=CONFIG.wandb.name,
        resume=False,
        config=CONFIG,
    )
    RUN_ID = wprj.id
else:
    RUN_ID = utils.get_random_hash()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, test_loader, train_test_loader = generate(
    CONFIG.dataset.batch_size, CONFIG.dataset.num_workers, device
)

mdl = getattr(models, CONFIG.model.name)
model = mdl()
model.to(device)
print(model)

args = {}
for an, av in CONFIG.hebbian.strategy.arguments.items():
    args[an] = av
trainer = Trainer(
    model=model,
    strategy=getattr(strategies, CONFIG.hebbian.strategy.name)(**args),
    num_classes=CONFIG.num_classes,
    lr=CONFIG.hebbian.learning_rate,
)

# visualize_feature_maps(
#    model=model,
#    samples=[train_loader.dataset.__getitem__(i) for i in range(10)],
#    device=device,
#    prefix="start",
# )

if model.has_hebbian():
    for i in range(CONFIG.hebbian.epochs):
        print(f"├── Hebbian [{i+1}/{CONFIG.hebbian.epochs}]")
        for images, labels in tqdm(train_loader):
            trainer(images, labels)

        test_loops(train_loader=train_test_loader, test_loader=test_loader, model=model)

# visualize_feature_maps(
#    model=model,
#    samples=[train_loader.dataset.__getitem__(i) for i in range(10)],
#    device=device,
#    prefix="hebbian",
# )

if model.has_backprop():
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG.backprop.learning_rate)

    for i in range(CONFIG.backprop.epochs):
        print(f"├── Backpropagation [{i+1}/{CONFIG.backprop.epochs}]")
        losses = []
        for images, labels in (pbar := tqdm(train_loader)):
            optimizer.zero_grad()
            y = model(images)
            loss = loss_fn(y, labels)
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
            pbar.set_description_str(f"├── {mean(losses):.4f}")

    test_loops(train_loader=train_test_loader, test_loader=test_loader, model=model)

# visualize_feature_maps(
#    model=model,
#    samples=[train_loader.dataset.__getitem__(i) for i in range(10)],
#    device=device,
#    prefix="backprop",
# )
