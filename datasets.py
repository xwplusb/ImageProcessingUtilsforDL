import os
from math import ceil
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class ImageData(Dataset):

    def __init__(self, root, image_size=256, transform=None):

        self.root = root
        self.image_size = (image_size, image_size)
        self.images = os.listdir(self.root)
        self.images = [os.path.join(root, image) for image in self.images if
                       image.endswith('jpg') or image.endswith('png')]

        self.transform = transform

        self.crop = transforms.RandomCrop(image_size)

    def __getitem__(self, item):
        image = self.images[item]
        image = read_image(image, ImageReadMode.RGB)
        _, h, w = image.size()
        if min(h, w) < min(self.image_size):
            scale = min(self.image_size) / min(h, w)
            resize = transforms.Resize(size=(ceil(h * scale), ceil(w * scale)))
            image = resize(image)

        image = self.crop(image)
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)


def getImageLoader(root, batch_size, image_size=256, transform=None):
    d = ImageData(root, image_size, transform)
    dd = DataLoader(d, batch_size, shuffle=True, num_workers=4)
    return dd


# if __name__ == '__main__':
#     dd = getImageLoader('coco_val2017', 4)
#     for i in dd:
#         print(i.shape)

