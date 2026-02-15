from svg_dataset import SVGDataset

dataset = SVGDataset("dataset/samples")

print(f"Nombre de samples : {len(dataset)}")
print()
print(dataset.samples[0])
print()
print(dataset[0][:200])