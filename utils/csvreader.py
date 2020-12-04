import csv

# Loads CSV files
def open_csv(path):
    x = []
    y = []
    with open(path, newline='') as admissionsfile:
        reader = csv.reader(admissionsfile)
        next(reader, None) # skip header line
        for row in reader:
            # Cast & add to array
            x.append([float(i) for i in row[1:8]])
            y.append(float(row[8]))
    
    return x, y
