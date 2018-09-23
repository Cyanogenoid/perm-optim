import math

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as T


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )


class SortNumbers(torch.utils.data.Dataset):
    def __init__(self, samples, low, high, size, double=False):
        self.samples = torch.Size([1, samples])
        self.low = low
        self.high = high
        self.size = size
        self.dtype = torch.float if not double else torch.double

    def __getitem__(self, item):
        samples = torch.rand(*self.samples, dtype=self.dtype) * (self.high - self.low) + self.low
        sorted_samples, idx = samples.sort()
        return samples, sorted_samples, idx.squeeze(0), 0

    def __len__(self):
        return self.size


class Mosaic(torch.utils.data.Dataset):
    def __init__(self, dataset, num_tiles, transform, image_size=None):
        self.dataset = dataset
        self.num_tiles = num_tiles
        self.transform = transform
        if not image_size:
            image_size = self.dataset[0][0].width
        self.tile_size = self.find_tile_size(image_size)
        self.crop = transforms.CenterCrop(self.tile_size * self.num_tiles)

    def find_tile_size(self, image_size):
        ratio = image_size / self.num_tiles
        return int(math.ceil(ratio))

    def __getitem__(self, item):
        img, label = self.dataset[item]
        # make sure image size is divisible
        if self.num_tiles * self.tile_size != img.width:
            img = T.resize(img, self.tile_size * self.num_tiles)
        if img.width != img.height:
            img = self.crop(img)
        tiles = []
        for i in range(self.num_tiles):
            y = i * self.tile_size
            for j in range(self.num_tiles):
                x = j * self.tile_size
                tile = T.crop(img, x, y, self.tile_size, self.tile_size)
                tile = self.transform(tile)
                tiles.append(tile)
        tiles = torch.stack(tiles, dim=0)
        perm = torch.randperm(tiles.size(0))  # sorted to unsorted
        reverse_perm = torch.zeros(tiles.size(0)).long()  # unsorted to sorted
        reverse_perm[perm] = torch.arange(tiles.size(0)).long()

        permuted_tiles = tiles[perm].transpose(0, 1)
        tiles = tiles.transpose(0, 1)
        return permuted_tiles, tiles, reverse_perm, label

    def __len__(self):
        return len(self.dataset)
