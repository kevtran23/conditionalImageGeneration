import numpy as np

embeddings_dict = {}
with open("glove.840B.300d.txt", 'r') as f:
    i = 0
    for line in f:
        values = line.split()
        word = values[0]    
        for j in range(1,len(values)):
            if(values[j][0].isdigit() == False):
                values[j] = '0.0' 
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
        i = i + 1

print(embeddings_dict)