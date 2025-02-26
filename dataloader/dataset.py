from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class SaliencyDataset(Dataset):

    def __init__(self, train_dir, label_dir, transform=None):

        self.train_dir = train_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.train_dir)

    def __getitem__(self, idx):

        train_img = cv2.imread(self.train_dir[idx])
        label_img = cv2.imread(self.label_dir[idx], 0)

        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((320, 240)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        label_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((320, 240)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        if self.transform:
            train_img = train_transform(train_img)
            label_img = label_transform(label_img)

        return train_img, label_img
