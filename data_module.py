import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from config import *
from preprocessing import preprocess

# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import RobertaTokenizer

# Set a random seed for reproducibility
pl.seed_everything(random_state, workers=True)


class SARCDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        texts = X

        texts = [preprocess(text) for text in tqdm(texts, desc="Preprocessing")]

        self._print_random_samples(texts)

        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=150,
                truncation=True,
                return_tensors="pt",
            )
            for text in tqdm(texts, desc="Tokenizing")
        ]

        self.labels = y

    def _print_random_samples(self, texts):
        print("Random samples after preprocessing:")
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, "labels"):
            label = self.labels[idx]

        return text, label


class SarcasmDetectionDataModule(pl.LightningDataModule):
    def __init__(self, data_file, batch_size=8, num_workers=0, mode="train"):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sarcasm_df = None
        self.mode = mode
        self.dataset = None

        self.prepare_data()

    def prepare_data(self):
        # print("Preparing data...")
        # sarcasm_df = pd.read_csv(self.data_file)

        # # We just need comment & label columns
        # # So, let's remove others.
        # sarcasm_df.drop(
        #     [
        #         "author",
        #         "subreddit",
        #         "score",
        #         "ups",
        #         "downs",
        #         "date",
        #         "created_utc",
        #         "parent_comment",
        #     ],
        #     axis=1,
        #     inplace=True,
        # )

        # print("Removing empty rows...")
        # # remove empty rows
        # sarcasm_df.dropna(inplace=True)

        # # Some comments are missing, so we drop the corresponding rows.
        # sarcasm_df.dropna(subset=["comment"], inplace=True)

        # # Calculate the lengths of comments
        # comment_lengths = [len(comment.split()) for comment in sarcasm_df["comment"]]

        # # Calculate the mean, maximum, and minimum lengths
        # mean_length = sum(comment_lengths) / len(comment_lengths)
        # max_length = max(comment_lengths)
        # min_length = min(comment_lengths)

        # # Print the results
        # print("Mean length:", mean_length)
        # print("Maximum length:", max_length)
        # print("Minimum length:", min_length)

        # print("Removing comments with length > 50...")
        # # Filter the dataframe to keep only comments with length <= 50
        # mask = [length <= 50 for length in comment_lengths]
        # sarcasm_df = sarcasm_df[mask]

        # # Reset the index of the dataframe
        # sarcasm_df.reset_index(drop=True, inplace=True)

        # self.sarcasm_df = sarcasm_df
        pass

    def setup(self, stage=None):
        print("Setting up data...")
        # print("Value counts:", self.sarcasm_df["label"].value_counts())

        # X_train, X_test, y_train, y_test = train_test_split(
        #     self.sarcasm_df["comment"],
        #     self.sarcasm_df["label"],
        #     test_size=test_size,
        #     random_state=random_state,
        # )

        # train_dataset = SARCDataset(X_train, y_train, tokenizer)
        # test_dataset = SARCDataset(X_test, y_test, tokenizer)

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.dataset = SARCDataset(
            self.sarcasm_df["comment"], self.sarcasm_df["label"], tokenizer
        )

        # Load preprocess/tokenized dataset using pickle
        # with open("preprocessed_dataset.pkl", "rb") as f:
        #     self.dataset = pickle.load(f)

        # Split the dataset into train and test set
        total_size = len(self.dataset)
        train_size = int(0.8 * total_size)  # 80% train, 20% test
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(
            self.dataset, [train_size, test_size]
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
