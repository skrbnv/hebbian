import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import libs.hbb as hbb


def visualize_receptive_field(model, input_shape, device):
    # Create a dummy tensor with the desired input shape
    dummy_input = torch.zeros(1, *input_shape).to(device)

    # Get the output shape of the last convolutional layer
    conv_output_shape = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_output_shape = module(dummy_input).shape

            # Compute the receptive field size
            receptive_field_size = (
                input_shape[1] / conv_output_shape[2],
                input_shape[2] / conv_output_shape[3],
            )

            # Compute the center of the receptive field
            center = (conv_output_shape[2] // 2, conv_output_shape[3] // 2)

            # Create a heatmap grid for visualization
            heatmap = torch.zeros(*input_shape[1:])

            # Iterate over each position in the heatmap and mark the corresponding receptive field center
            for i in range(input_shape[1]):
                for j in range(input_shape[2]):
                    x_offset = int((i - center[0]) / receptive_field_size[0])
                    y_offset = int((j - center[1]) / receptive_field_size[1])
                    if (
                        0 <= x_offset < conv_output_shape[2]
                        and 0 <= y_offset < conv_output_shape[3]
                    ):
                        heatmap[i, j] = 1

            # Plot the heatmap
            plt.imshow(heatmap, cmap="hot", interpolation="nearest")
            plt.title("Receptive Field Visualization")
            plt.colorbar()
            plt.savefig("preview.png")
            plt.close()
            break


def visualize_filters(model, input_shape, device, prefix=""):
    x = torch.zeros(1, *input_shape).to(device)

    for layer_num, m in enumerate(model.modules()):
        if isinstance(m, hbb.Conv2d):
            x = m(x)
            w = m.weight.data
            w -= w.min()
            w /= w.max()
            rows, cols = w.size(0), w.size(1)
            combined = torch.ones(
                (rows * w.size(2) + rows - 1, cols * w.size(3) + cols - 1)
            )
            plt.figure(figsize=(cols, rows))
            for row in range(rows):
                for col in range(cols):
                    combined[
                        row * w.size(2) + row : row * w.size(2) + row + 3,
                        col * w.size(3) + col : col * w.size(3) + col + 3,
                    ] = w[row, col]
                    # axs[row, col].imshow(w[row, col].cpu(), cmap="gray")
            plt.axis("off")
            plt.imshow(combined, cmap="gray")
            plt.savefig(f"{prefix}{layer_num}.png")

            # Plot the heatmap
            # plt.imshow(heatmap, cmap="hot", interpolation="nearest")
            # plt.title("Receptive Field Visualization")
            # plt.colorbar()
            # plt.savefig("preview.png")
            # plt.close()
            # break


def visualize_feature_maps(model, samples, device, prefix=""):
    for i, (sample, _) in enumerate(samples):
        x = sample.unsqueeze(0)
        x -= x.min()
        x /= x.max()
        plt.figure(figsize=(5, 5))
        plt.imshow(sample.permute(1, 2, 0).cpu())
        plt.savefig(f"sample{i}.png")
        plt.close()

    for layer_num, m in enumerate(model.modules()):
        if isinstance(m, hbb.Conv2d):
            x = m(x)
            rows, cols = x.size(0), x.size(1)
            combined = torch.ones(
                (rows * x.size(2) + rows - 1, cols * x.size(3) + cols - 1)
            )
            plt.figure(figsize=(cols, rows))
            for row in range(rows):
                for col in range(cols):
                    combined[
                        row * x.size(2) + row : row * x.size(2) + row + x.size(2),
                        col * x.size(3) + col : col * x.size(3) + col + x.size(3),
                    ] = x[row, col]
                    # axs[row, col].imshow(w[row, col].cpu(), cmap="gray")
            plt.axis("off")
            plt.imshow(combined)
            plt.savefig(f"{prefix}{layer_num}.png")
            plt.close()

            # Plot the heatmap
            # plt.imshow(heatmap, cmap="hot", interpolation="nearest")
            # plt.title("Receptive Field Visualization")
            # plt.colorbar()
            # plt.savefig("preview.png")
            # plt.close()
            # break
