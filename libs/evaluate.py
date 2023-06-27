import torch
from tqdm import tqdm


def test_loop(loader, model):
    correct, total = 0, 0
    for inputs, labels in (pbar := tqdm(loader)):
        with torch.no_grad():  # .flatten(1)
            x = model(inputs)
            correct += torch.argmax(x, dim=1).eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_description_str(
                f"├── C: {correct}, T: {total}, Acc: {correct*100/total:.2f}%"
            )
    return correct * 100 / total
