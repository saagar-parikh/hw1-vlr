import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils import get_data_loader
from train_q2 import ResNet


def main():
    device = "cuda"

    # Load the model
    model = torch.load("checkpoint-model-epoch9.pth")
    # https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
    model = nn.Sequential(*list(model.children())[:-1]).to(device)
    model.eval()

    # Load the data
    data_loader = get_data_loader(
        "voc", train=False, batch_size=1, split="test", inp_size=224
    )

    # Get the features
    features = []
    targets = []

    for i, (img, target, j) in enumerate(data_loader):
        with torch.no_grad():
            # Get the features from the model (penultimate layer)
            output = model(img.to(device))
        features.append(output)
        targets.append(target)
        if i == 999:
            break

    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)

    features = features.view(features.size(0), -1).cpu().numpy()
    targets = targets.cpu().numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=1234)
    features = tsne.fit_transform(features)

    # Plot the features
    plt.figure(figsize=(10, 8))
    for i in range(20):
        # Get the indices of the current class
        indices = np.where(targets[:, i] == 1)[0]
        plt.scatter(features[indices, 0], features[indices, 1], label="Class " + str(i))

    plt.title("t-SNE Projection of Features")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("tsne.png")


if __name__ == "__main__":
    main()
