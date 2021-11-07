import datasets
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import OrderedDict
from pathlib import Path
from torchtext.legacy.data import Field
from tqdm import tqdm

work_dir = Path(__file__).parent.parent.resolve()
data_dir = work_dir / "data"

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


dataset_name = "bible_para"

train_data = datasets.load_dataset(dataset_name, "de-en", split="train", cache_dir=str(data_dir))[:-5000]
valid_data = datasets.load_dataset(dataset_name, "de-en", split="train", cache_dir=str(data_dir))[-5000:-2500]
test_data = datasets.load_dataset(dataset_name, "de-en", split="train", cache_dir=str(data_dir))[-2500:]

inputs = "de"
labels = "en"


def get_values_list(some_list_of_dicts, key):
    return [i[key] for i in some_list_of_dicts]

# train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir=str(data_dir))
# valid_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation", cache_dir=str(data_dir))
# test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test", cache_dir=str(data_dir))[:100]
#
# inputs = "article"
# labels = "highlights"


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)


print("Building the vocabulary")
# SRC.build_vocab(tqdm([tokenize_en(i) for i in train_data["article"]]), min_freq=2)

SRC.build_vocab(tqdm([tokenize_de(i) for i in get_values_list(train_data["translation"], inputs)]), min_freq=1)
TRG.build_vocab(tqdm([tokenize_en(i) for i in get_values_list(train_data["translation"], labels)]), min_freq=1)

print(len(SRC.vocab), len(TRG.vocab))


def stoi_list(sents, field):
    sents_indices = []

    for sent in sents:
        sents_indices.append([field.vocab.stoi[i] for i in sent])

    sents_tensors = []
    max_length = len(max(sents_indices, key=len))

    for sent_indices in sents_indices:
        indices_tensor = torch.tensor(sent_indices, dtype=torch.int64)
        pad_tensor = torch.tensor([field.vocab.stoi[field.pad_token] for i in range(max_length - len(sent_indices))],
                                  dtype=torch.int64)
        tensor = torch.cat((indices_tensor, pad_tensor))
        sents_tensors.append(tensor)

    stack = torch.stack(sents_tensors)
    return stack


def collate_batch(batch):
    # print(f'here is the batch: {batch}')
    src_list = [i['src'] for i in batch]
    trg_list = [i['trg'] for i in batch]
    # print(f'src list: {src_list}')
    # print(f'trg list: {trg_list}')
    src_tensors = stoi_list(src_list, SRC)
    trg_tensors = stoi_list(trg_list, TRG)
    return dict(src=src_tensors, trg=trg_tensors)


class CustomDataSet(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        src = self.source[idx]
        src_tokens = tokenize_de(src)
        trg = self.target[idx]
        trg_tokens = tokenize_en(trg)
        sample = {"src": src_tokens, "trg": trg_tokens}
        return sample


train_dataset = CustomDataSet(get_values_list(train_data["translation"], inputs), get_values_list(train_data["translation"], labels))
valid_dataset = CustomDataSet(get_values_list(valid_data["translation"], inputs), get_values_list(valid_data["translation"], labels))
test_dataset = CustomDataSet(get_values_list(test_data["translation"], inputs), get_values_list(test_data["translation"], labels))

# train_dataset = CustomDataSet(train_data[inputs], train_data[labels])
# valid_dataset = CustomDataSet(valid_data[inputs], valid_data[labels])
# test_dataset = CustomDataSet(test_data[inputs], test_data[labels])

phrase = ["Am", "Anfang", "schuf", "Gott", "Himmel", "und", "Erde"]

test_iterator = DataLoader(test_dataset, batch_size=2, collate_fn=collate_batch, shuffle=True, pin_memory=True)

if __name__ == '__main__':
    print(train_data["translation"][0])
    print(train_dataset[0])
    print([SRC.vocab.stoi[i] for i in phrase])
    for (idx, batch) in enumerate(test_iterator):
        print(idx, 'SRC data: ', batch["src"])
        print(idx, 'TRG data: ', batch["trg"], '\n')
