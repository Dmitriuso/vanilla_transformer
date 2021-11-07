import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from pathlib import Path

import numpy as np
import random
import math
import time
import pkbar

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq

from utils.pre_processing_sum_spacy import train_dataset, valid_dataset, test_dataset, collate_batch, SRC, TRG
from inference_sum_spacy import SummaryInference

work_dir = Path(__file__).parent.resolve()
models_dir = work_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

data_dir = work_dir / "data"

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_model_path = models_dir / "translate_custom_spacy.pt"

BATCH_SIZE = 64

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True, num_workers=8)
valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True, num_workers=8)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True, num_workers=8)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 1024
DEC_PF_DIM = 1024
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(output_dim=OUTPUT_DIM,
              hid_dim=HID_DIM,
              n_layers=DEC_LAYERS,
              n_heads=DEC_HEADS,
              pf_dim=DEC_PF_DIM,
              dropout=DEC_DROPOUT,
              device=device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip, kbar):
    model.train()

    epoch_loss = 0

    for (i, batch) in enumerate(iterator):
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        # print(f'src in batch: {src.shape}')
        # print(f'trg in batch {trg.shape}')

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # print('-' * 50)
        # print(f'trg : {trg.shape}')
        # print(f'output : {output.shape}')
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        kbar.update(i, values=[("loss", epoch_loss / float(i + 1))])

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for (i, batch) in enumerate(iterator):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    N_EPOCHS = 15
    CLIP = 1

    best_valid_loss = float('inf')
    writer = SummaryWriter()

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        kbar = pkbar.Kbar(target=len(train_iterator), epoch=epoch, num_epochs=N_EPOCHS, width=8, always_stateful=False)

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, kbar)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, output_model_path)

        print(f'\n\tEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')
        writer.add_scalar("Train Loss", train_loss, epoch+1)
        writer.add_scalar("Train PPL", math.exp(train_loss), epoch+1)
        writer.add_scalar("Valid Loss", valid_loss, epoch+1)
        writer.add_scalar("Valid PPL", math.exp(valid_loss), epoch+1)

        model_test_path = output_model_path
        model_test = torch.load(model_test_path)
        test_loss = evaluate(model_test, test_iterator, criterion)
        print(f'\tTest Loss: {test_loss:.3f}  | Test PPL: {math.exp(test_loss):7.3f}')

    example_idx = 7

    train_src = train_dataset[example_idx]["src"]
    train_trg = train_dataset[example_idx]["trg"]

    val_src = valid_dataset[example_idx]["src"]
    val_trg = valid_dataset[example_idx]["trg"]

    test_src = test_dataset[example_idx]["src"]
    test_trg = test_dataset[example_idx]["trg"]

    rand_article = "Ich habe aber alles verstanden" # open(str(data_dir / "article.txt"), "rt").read()

    summary = SummaryInference(
        model_path=output_model_path,
        src_field=SRC,
        trg_field=TRG,
        max_len=50,
        device=device,
    )

    train_summary = summary.inference(train_src)
    val_summary = summary.inference(val_src)
    test_summary = summary.inference(test_src)

    print(f"train src = {train_src}")
    print(f"train trg = {train_trg}")
    print(f"predicted train trg = {train_summary}")

    print(f"val src = {val_src}")
    print(f"val trg = {val_trg}")
    print(f"predicted val trg = {val_summary}")

    print(f"test src = {test_src}")
    print(f"test trg = {test_trg}")
    print(f"predicted test trg = {test_summary}")

    print(f"arbitrary example: {rand_article}")
    print(f"prediction: {summary.inference(rand_article)}")


if __name__ == '__main__':
    main()
