from svg_dataset import SVGDataset

dataset = SVGDataset("dataset/samples")

print(f"Nombre de samples : {len(dataset)}\n")
print(dataset.samples[0],"\n")
print(dataset[0][:500])