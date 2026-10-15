import numpy as np, camp as cp
rng = np.random.default_rng(1); D = 512
persons = ["ion","ana","dan","maria","radu","ioana","vlad","elena"]; objects = ["carte","mar","pix","bani","ceas","tort","cheie","pahar"]
w = cp.World(persons, objects, rng)
wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)
def pq(m, n=200):
    cs=[]
    for _ in range(n):
        ev=w.event(); t=rng.integers(5); cs.append(m.cos(m.state(w.render(ev,t))*m.T, wm.state(w.cons(ev))))
    return np.mean(cs)
def confusion(m, kind="GIVE", t=1, n=200):
    addr = m.role_addresses(kind); swap=0; 
    for _ in range(n):
        ev=w.event(); ev=(kind,)+ev[1:]; S=m.state(w.render(ev,t))
        swap += m.decode(S, addr["C"], persons) == ev[1]
    return swap/n
m = cp.Model(w, D, np.random.default_rng(2), deferred=True)
for stage, (n, lr) in enumerate([(12000, 2.0), (12000, 2.0), (24000, 4.0)]):
    m.train(wm, n_sent=n, lr=lr)
    res = m.evaluate(n=100); allv=[v for a in res.values() for v in a.values()]
    print(f"stage {stage}: predictie={pq(m):.3f} media decodare={np.mean(allv):.3f}  T1 dest decodat ca AGENT in {confusion(m):.2f} din cazuri; GIVE T1: {res[('GIVE',1)]}")
