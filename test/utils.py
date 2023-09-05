import csv


def read_slices_from_csv(csv_file):
    slices = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start = int(row["Start"])
            end = int(row["End"])
            slices.append(slice(start, end))

    return slices
