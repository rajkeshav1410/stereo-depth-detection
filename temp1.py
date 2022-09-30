import pickle

# with open('names', 'ab') as f:
#     pickle.dump(l, f)

with open('names', 'rb') as f:
    l = pickle.load(f)
    print(l)