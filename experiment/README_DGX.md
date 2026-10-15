# Plan de validare pe DGX Spark

## Ce ai in zip
- camp.py, k.py: modelul sintetic complet (pas uniform, adresare amanata, zoom-in, invatare generativa) + evaluare pe roluri
- control_transformer.py: controlul (transformer numpy) pe aceeasi sarcina
- sick_test.py, wikisql_test.py: primele teste pe text real (negativ pe SICK, modest-pozitiv pe WikiSQL)
- REZULTATE.md: toate numerele si concluziile; log_*.txt: iesiri brute; corp/: SICK (mic) — WikiSQL se descarca (vezi mai jos)

## Portare in torch (prima zi)
Totul e numpy cu gradiente scrise de mana ca sa mearga aici fara torch. In torch:
  theta = nn.Parameter(...); m = torch.polar(torch.ones_like(theta), theta)   # deplasari ca unitati complexe
  L = torch.cumprod(m[cats], dim=0); S = (L * F[ids]).sum(0)                  # pasul uniform, pe GPU, in batch
  loss = -cosine(S * T, S_lume)                                                # autograd face restul
Batch: grupeaza propozitiile pe lungime (sau pad + masca). D=4096-16384 incape lejer in memoria unificata.

## Experimentele, in ordinea valorii (fiecare cu criteriul de "merge / nu merge")
1. WikiSQL complet (56k; 15k test) cu: categorii descoperite (clustering iterativ ca in i2.py), adresare amanata cu declansator
   invatat (vezi 3), D=8192. Merge daca: col.selectata > 0.7 si agregator > 0.85 FARA sa vada tabelul; si perechi acelasi-schelet
   separate de schelet-diferit cu > 0.3 cosinus. Nu merge daca ramane sub sacul de cuvinte + regresie logistica pe aceiasi tokeni (fa-l ca baseline!).
2. Spider (github.com/taoyds/spider; mai greu, SQL imbricat): testeaza zoom-in-ul (k.py) pe sub-interogari.
3. Invatarea declansatoarelor: in loc de "verbul scrie bufferul", invata un scor per cuvant/categorie care decide flush-ul (sigmoid, gradient prin
   bundle ponderat: S += sigma(s_w) * termen_amanat + (1 - sigma(s_w)) * termen_imediat). Merge daca scorul invatat se aliniaza cu verbele/
   cuvintele de legatura fara sa i le fi spus.
4. Echivalenta cu negative grele: perechi (intrebare, intrebare cu rolurile/valorile schimbate) construite automat din SQL; raporteaza separat
   acuratetea pe aceste perechi, nu doar corelatia globala (SICK a aratat ca media ascunde exact ce conteaza).
5. Control puternic: un transformer mic in torch (sau un BERT mic) pe aceeasi sarcina si aceleasi date; compara la date putine (1k, 5k, 56k).
6. Capacitate: coduri rare pe blocuri (sparse block codes; Laiho/Frady/Kleyko) in loc de faze dense, pentru vocabular de 50k+.

## Surse de date cu echivalenta "data de lume" (toate publice)
- WikiSQL: https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2 (26 MB; jsonl + sqlite)
- Spider (text->SQL imbricat): https://github.com/taoyds/spider (link catre Google Drive in README)
- CoSQL / SParC: conversatii -> SQL (yale-lily)
- Cod + teste: APPS, CodeContests, MBPP (HuggingFace datasets); echivalenta = aceleasi teste trecute
- Demonstratii: Lean mathlib (github.com/leanprover-community/mathlib4), miniF2F; linii de derivare = echivalente prin constructie
- Instructiuni cu stare: ProPara (github.com/allenai/propara), ALFRED (askforalfred.com), TextWorld (Microsoft)
- Legende multiple per imagine: COCO captions (images.cocodataset.org/annotations/annotations_trainval2017.zip), Flickr30k (Kaggle)
- Traduceri paralele: OPUS (opus.nlpl.eu), Tatoeba (tatoeba.org/downloads) — adauga negative grele automat
- Parafraze cu negative grele gata facute: PAWS (github.com/google-research-datasets/paws), PAWS-X (6 limbi, inclusiv ... nu romana; dar OPUS are)
- Markeri de reformulare din manuale/Wikipedia ("adica", "cu alte cuvinte", "i.e."): extragere proprie, volum mare, zgomot mare

## La ce te poti astepta, realist
- Nu la un model care bate LLM-urile pe text->SQL (acelea sunt la ~90%; aici, fara tabel si cu 10k parametri, am 0.49 dupa 70 s de CPU).
- La un strat de stare semantica citibila (cine-ce-cui), antrenabil cu putine date, care generalizeaza la cuvinte noi fara reantrenare
  si care poate VERIFICA echivalenta a doua formulari rol cu rol. Util ca verificator/interpret langa un LLM, in domenii inchise.
- Cel mai valoros rezultat stiintific posibil: daca declansatoarele (3) se invata singure pe WikiSQL/Spider, atunci intreg mecanismul
  (categorii, cadre, roluri, ierarhie) e emergent dintr-un singur obiectiv de predictie generativa. Asta ar merita publicat.

## Dupa runda 2: ce sa rulezi intai pe DGX
0. Reproduce tabelul principal din REZULTATE.md "RUNDA 2" cu 5 seed-uri (v2.py, v3.py), apoi gates.py cu N mare.
1. Generalizare la CONSTRUCTII noi (bariera reala): lume cu 10-20 formulari; tine deoparte formulari ale caror tranzitii au fost toate vazute
   separat vs formulari cu tranzitii noi. Testeaza coduri REDUNDANTE pentru adrese (bundle de mai multe drumuri / blocuri), ca o tranzitie
   neantrenata sa nu randomizeze toata adresa.
2. Porti invatate prin gradient (sigmoid + amestec moale intre adresa provizorie si cea asteptata) in loc de cautare discreta; verifica
   daca converg la acelasi set (identitatile inchise, verbele/prepozitiile deschise).
3. WikiSQL cu v3 (dual) + porti invatate + categorii descoperite; raporteaza marja, nu doar acuratetea.
