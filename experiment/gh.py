import numpy as np, camp as cp
rng = np.random.default_rng(1); D = 512
persons = ["ion","ana","dan","maria","radu","ioana","vlad","elena","mihai","irina","paul","sofia"]
objects = ["carte","mar","pix","bani","ceas","tort","cheie","pahar","lampa","scaun","harta","inel"]
train_P, test_P = persons[:8], persons[8:]; train_O, test_O = objects[:8], objects[8:]
w = cp.World(persons, objects, rng)
wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)
m = cp.Model(w, D, np.random.default_rng(2), deferred=True)
for stage in range(5):
    m.train(wm, n_sent=10000, lr=0.02, persons=train_P, objects=train_O)
    res = m.evaluate(n=100, persons=train_P, objects=train_O); allv=[v for a in res.values() for v in a.values()]
    print(f"G2' (buffer cu drum relativ): dupa {(stage+1)*10000} propozitii, media = {np.mean(allv):.3f}", flush=True)
print("\n=== G2' pe identitati VAZUTE (8 persoane, 8 obiecte; sansa 1/8)"); print(cp.summarize(m.evaluate(n=150, persons=train_P, objects=train_O), w))
print("\n=== H: aceleasi ponderi, identitati NEVAZUTE la antrenare (4 persoane, 4 obiecte noi; sansa 1/4)"); print(cp.summarize(m.evaluate(n=150, persons=test_P, objects=test_O), w))
print("\n=== H': toate 12+12, amestecate (sansa 1/12)"); print(cp.summarize(m.evaluate(n=150), w))
import pickle; pickle.dump(m, open("model_G2.pkl","wb"))
