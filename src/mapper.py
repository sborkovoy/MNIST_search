import sys

import io
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import torch


class ContrastiveAutoencoder(nn.Module):
    def __init__(self, emb_dim=2):
        super(ContrastiveAutoencoder, self).__init__()
        self.emb_dim = emb_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=5),
            nn.Tanh()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        return self.encoder(x)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def simple_similarity_search(test_image, model, train_embeddings, train_labels, k=5):
    """
    Args:
        test_image: тензор [1, 1, 28, 28]
        model: обученная модель
        train_embeddings: numpy array [num_train, emb_dim]
        train_images: numpy array [num_train, 1, 28, 28] или список
        train_labels: numpy array [num_train]
        k: число соседей
    """
    model.eval()
    with torch.no_grad():
        test_emb = model.get_embedding(test_image).cpu().numpy()

    distances = np.linalg.norm(train_embeddings - test_emb, axis=1)
    nearest_idx = np.argsort(distances)[:k]

    for i, idx in enumerate(nearest_idx):
        print(f"Label: {train_labels[idx]}\nDist: {distances[idx]:.3f}")

    return nearest_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(ContrastiveAutoencoder(emb_dim=2), 'contrastive_autoencoder_mnist.pth')

all_embeddings = np.load("./all_embeddings.npy")
all_labels = np.load("./all_labels.npy")

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),                # -> (1, 28, 28), значения [0, 1]
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize((0.5,), (0.5,))  # как в обучении MNIST, если так было
])

while True:
    data = tb.read(1)  # data
    if data is None:
        break

    image = Image.open(io.BytesIO(data)).convert('RGB')
    tensor_image = transform(image)
    try:
        nearest = simple_similarity_search(tensor_image.unsqueeze(0), model, all_embeddings, all_labels, k=5)
    except Exception as e:
        sys.stderr.write(f"Error {e}\n")