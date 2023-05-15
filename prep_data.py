import csv
import random

num_examples = 100
filename = "test"

filename = filename + ".csv"

# Generate random text and labels
data = [["text", "label"]]
for i in range(num_examples):
    text = "This is example text number " + str(i)
    label = random.choice([0, 1])  # Random binary label
    data.append([text, label])

# Write the data to a CSV file
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
