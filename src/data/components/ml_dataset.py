import os
import shutil
import zipfile
import gdown

import numpy as np 
import pandas as pd 
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

from typing import Any

class MLDataset(Dataset):
    def __init__(
            self,
            data_dir: str = "data",
            is_train: bool = True,
            text_pretrained : str = "bert-base-uncased",
            transform: Any = None,
    ) -> None:
        super.__init__()

        self.data_dir = data_dir
        self.text_pretrained = text_pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(text_pretrained)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

        self.prepare_data()

        folder_img_path = os.path.join(self.data_dir, "ml1m/content/dataset/ml1m-images")

        if is_train:
            self.data = pd.read_csv(os.path.join(self.data_dir, "ml1m/content/dataset/movies_train.dat"), engine='python',
                                    sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
        else:
            self.data = pd.read_csv(os.path.join(self.data_dir, "ml1m/content/dataset/movies_test.dat"), engine='python',
                                    sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
        
        self.data['genre'] = self.data.genre.str.split('|')

        # Add img_path to dataframes and remove rows with no images
        self.data['id'] = self.data.index
        self.data.reset_index(inplace=True)
        self.data['img_path'] = self.data.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)

        self.data['img_exists'] = self.data['img_path'].apply(lambda x: os.path.exists(x))
        self.data = self.data[self.data['img_exists']]
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.drop(columns=['img_exists'])

        # Filter out year from titles as it seems to be unnecessary
        for i in range(len(self.data)):
            self.data['title'].iloc[i] = re.sub('\(\d{0,4}\)', '', self.data['title'].iloc[i])

        with open(os.path.join(self.data_dir, "ml1m/content/dataset/genres.txt"), 'r') as f:
            genre_all = f.readlines()
            genre_all = [x.replace('\n','') for x in genre_all]
        self.genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        title = self.data['title'].iloc[index]
        img_path = self.data['img_path'].iloc[index]
        genres = self.data['genre'].iloc[index]

        title = self.tokenizer(title, padding='max_length', return_tensors='pt')
        title = {k: torch.squeeze(v, dim=0) for k, v in title.items()}

        labels = torch.zeros((len(self.genre2idx)), dtype=torch.float)
        for genre in genres:
            labels[self.genre2idx[genre]] = 1

        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        transformed = self.transform(image=image)
        image = transformed["image"]

        return (image, title, labels)
    
    def prepare_data(self):
        data_path = os.path.join(self.data_dir, "ml1m")
        if os.path.exists(data_path):
            print("Data is downloaded")
            return
        
        file_id = "1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD"
        output = "ml1m.zip"
        print("Downloading data")
        gdown.download(id=file_id, output=output, quiet=False)

        os.makedirs(data_path, exist_ok=True)

        shutil.move("./ml1m.zip", data_path)

        downloaded_file = os.path.join(data_path, "ml1m.zip")

        print("Extracting ...")
        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(data_path)

        print("Removing unnecessary files and folders")
        os.remove(downloaded_file)  # delete zip file

        print("Done!")
    
if __name__ == "__main__":
    mldataset = MLDataset()
    img, title, label = mldataset[2]
    print(img.shape)
    print(title)
    print(label)
    print(label.shape)
                 