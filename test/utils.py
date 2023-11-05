import csv

import torch


def read_slices_from_csv(csv_file):
    slices = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start = int(row["Start"])
            end = int(row["End"])
            slices.append(slice(start, end))

    return slices


def count_bits(x) -> int:
    count = torch.zeros_like(x)
    while x.any():
        count += x & 1
        x >>= 1
    return count.item()
