# Jurnal de lucru (o intrare per sesiune; format in GIT_WORKFLOW.md §3)

## 2026-10-14/15 — sesiunea initiala (Claude + autor), CPU, numpy
Rulat: sim_deplasari.py, camp.py, k.py, control_transformer.py, mlp_control.py, sick_test.py, wikisql_test.py, v2.py, v3.py, gates.py, nest2.py, v4.py
Rezultat: vezi REZULTATE.md (3 runde). Pe scurt: factorizare + obiectiv generativ => roluri aliniate; porti invatabile; redundanta 0.60 -> 0.90.
Surpriza / problema: doua greseli de evaluare in runda 1 (adrese din forma activa; F diferit lume/model) au produs "reguli necesare" false.
Schimbat in cod: evaluarea refacuta (v2.evaluate: adrese din lume); PRINCIPII_EVALUARE.md scris.
Urmatorul pas: portare torch; v4 cu 5 seed-uri; WikiSQL cu modelul complet + baseline BoW+LR.

## 2026-10-15 (DGX) — curatenie documente: aliniere la runda 2-3, fara cod, fara rulari
Rulat: nimic (nicio cifra schimbata; sursa de adevar: 00_CITESTE_INTAI.md si REZULTATE.md runda 2-3; ce contrazice runda 2 e marcat RETRAS/DEPASIT, nu sters).
Schimbat: POZITIONARE.md (contributia (i) -> adrese suprapuse + argmax; fig.1 (b) -> coduri redundante; porti invatate = contributie noua;
  bariera "declansatoare de mana" marcata retrasa); README_DGX.md (plan vechi marcat DEPASIT; ramane trimitere + portare + surse de date);
  README_RUN.md (MODELUL FINAL = v4.py; camp.py = lumea + v1 istoric); REZULTATE.md runda 3 (nota: toate 4 0.94 < bigram 1.00 pe tranzitii vazute);
  00_CITESTE_INTAI.md §4 (imbricare si porti invatate mutate din SIGUR in PROBABIL); logs/README.md (loguri vechi = pre-run.sh); ANALIZA_INITIALA_DGX.md in radacina.
Urmatorul pas: mediu (venv, numpy, torch CUDA aarch64) si prima rulare noua ./run.sh v4.py pe GPU, in commit separat.

## 2026-10-15 (DGX, dupa-amiaza) — mediu, reproducere pe DGX, portare torch, batch
Rulat: run.sh v4.py (numpy CPU)            -> logs/2026-10-15_1101_v4_d26b7db.txt        (10 min; 24.8 s/varianta)
       v4_torch.py GPU batch=1              -> logs/2026-10-15_1107_v4_torch_c57d74a.txt  (lansat manual cu format run.sh: run.sh refuza
                                                arbore murdar, iar logul rularii CPU inca necomis il murdarea; de aici incolo secvential)
       run.sh v4_torch.py --batch 32 / 8    -> logs/2026-10-15_1119_v4_torch_f8fedc5.txt / ..._1120_v4_torch_fc97386.txt
Rezultat: numpy pe DGX reproduce EXACT tabelul rundei 3 (3 decimale, 8 variante). torch GPU batch=1 da scoruri identice cu numpy pe toate
  24 liniile (acuratete + marja): portarea e echivalenta numeric. Batch 8/32: toate 4 -> 0.885/0.887 pe tranzitii nevazute (vs 0.901),
  in marja +-0.03; codurile slabe (path, path+chain) pierd la batch 32. Tabel complet in REZULTATE.md "RUNDA 4".
Surpriza / problema: la D=512, batch=1, GPU-ul nu e mai rapid decat CPU-ul (28 s vs 25 s/varianta): cost = Python + pasi mici.
  Cu encodare o singura data + batch 32: 2.2 s/varianta (10x), din care 1 s e tot Python (encodarea). Roata torch e cea corecta (cu130, aarch64).
Schimbat in cod: v4_torch.py NOU (structura adreselor ramane din v2/v4 numpy; faze -> termeni -> cos -> autograd in torch; --batch, --lr,
  --seeds, --n_pers, --D, --device); .gitignore radacina (.venv). Mediu: .venv cu numpy 2.5.4, torch 2.14.1+cu130.
Urmatorul pas: 00_CITESTE_INTAI §5.1 — lume de 2-3x (4-6 cadre, 10-20 formulari, imbricare), 5 seed-uri, cu v4_torch batch 8-32;
  apoi §5.2 porti prin gradient.

## 2026-10-15 (DGX, seara) — §5.1: lume de 3x, 5 seed-uri (v5_scale.py) + diagnostic (diag_v5.py)
Rulat: run.sh v5_scale.py (5 seed-uri, 7 variante, D=1024, 30k, batch 8) -> logs/2026-10-15_1128_v5_scale_0defffd.txt (12.7 min)
       diag_v5.py + detaliere per formulare x rol si per cod la adancime 2 (seed 0; iesirile in REZULTATE.md runda 5)
Rezultat: path+chain+bigram REPRODUS intre lumi (0.998/0.821 vs 0.987/0.824). toate 4 NU: 0.841 / 0.984 (inversat fata de 0.942 / 0.901).
  Imbricare: ad.1 1.00; ad.2 (vazuta!) ~0.85, X2 0.45 confundat cu A; ad.3: X3 la sansa prin constructie (niciodata tinta).
Surpriza / problema: (1) pos = prior de pozitie absoluta: T1 e singurul tipar de antrenare cu persoana la poz. 5 (agent) => in T10 (destinatar
  la 5) pos voteaza complet gresit si bate adresele suprapuse ale primului cuvant. Castigul lui pe tranzitii nevazute vine din categorii noi
  ('catre', 'il') unde celelalte coduri n-au nimic. Ambele sunt proprietati ale setului de formulari.
  (2) MAI IMPORTANT: '{X1} spune ca {A} V..' si '{X1} spune ca {X2} zice ca..' au prefix identic ('P spune ca P') => orice adresa din prefix
  da acelasi loc pentru A (ad.1) si X2 (ad.2) => suprapunere ~0.5. Imbricarea >= 2 e nerezolvabila cu coduri cauzale, indiferent de D/date.
  "Adresarea amanata" retrasa in runda 2 revine, in alta forma: reasezarea unui cuvant de catre ce urmeaza. Prima dovada curata ca e necesara.
Schimbat in cod: v5_scale.py (BigWorld, BigRoleWorld, V5 cu pos pana la 40), diag_v5.py. Documente: REZULTATE runda 5; PRINCIPII 13-15
  (acelasi regim de optimizare; rol nou => raportare pe roluri; detaliere per formulare x rol inainte de interpretare); 00_CITESTE §4-5
  (imbricare >= 2 NEGATIV; §5.1 facut; §5.1b nou: adresare amanata invatata); POZITIONARE (bariera redeschisa partial).
Urmatorul pas: §5.1b (X2 la ad.2 > 0.9 fara pierderi) inainte de §5.2, pentru ca portile prin gradient depind de acelasi mecanism de asteptari.
  Intrebare pentru judecata: raportam 'modelul' ca path+chain+bigram (0.99/0.82) si pos separat ca prior? Eu zic da.
