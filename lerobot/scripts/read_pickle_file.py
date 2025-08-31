import pickle

with open("data/episode_f_0002.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)
