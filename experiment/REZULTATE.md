# Campanie de experimente: cuvintele ca deplasari, starea ca bundle de (adresa ⊗ identitate)

Punct de pornire: "fraze diferite, acelasi sens = drumuri diferite, acelasi punct" intr-un spatiu HDC/FPE
(vectori complecsi de modul 1, bind = inmultire, bundle = adunare).

## Modelul final (camp.py / k.py)
- Fiecare cuvant: **identitate** f_w (vector fix aleator) si **deplasare** m_cat(w) (invatata, una pe categorie).
- Pas uniform: L <- L * m ; S <- S + L * f.  Starea unei propozitii = S.
- **Adresare amanata**: cuvintele dinaintea primului verb sunt scrise abia la verb, la adresa r_verb ⊗ drum_relativ.
- **Zoom in**: subordonatorul "ca" fixeaza o ancora; drumul interior porneste de la zero relativ la ea.
- Invatare: doar eroare de predictie generativa, cos(S ⊗ T, S_lume(consecinta)) cu T fix aleator;
  starea lumii e "senzoriala" (cod fix). Fara etichete de parafraze, fara negative. Adam, lr 0.02.
- Evaluare: se citesc rolurile (agent, obiect, destinatar) din starea oricarei formulari folosind
  *adresele formei active*; corect = aceeasi identitate recuperata prin cleanup.

## Rezultate (decodare medie pe toate formularile si rolurile; sansa = 1/n identitati)

| exp | ce s-a testat | rezultat |
|---|---|---|
| A | predictie locala cuvant->cuvant | colaps: totul similar; roluri = pozitii sintactice |
| B | contrastiv pe parafraze (etichete) | 99.8% parafraza > schimb roluri; 70% pe perechi noi |
| C | predictia consecintei prin memorie asociativa | predictie 100%, dar NICIO aliniere a starilor |
| D | prototipuri per consecinta | separa obiectul, nu rolurile; nu generalizeaza |
| E | generativ S⊗T, ambele parti invatate | colaps |
| E2 | generativ, lume fixa, deplasare per cuvant | esec; demonstrat algebric imposibil |
| F | + deplasare pe categorie (ce face / ce este) | roluri aliniate pe pasiv (dest 1.00, obiect 1.00, agent 0.64); subiectul initial ramane pozitional |
| G2 | + adresare amanata la verb (5 formulari x 2 cadre, 8 id) | **0.86** (G1 fara amanare 0.70; G3 fara factorizare 0.50) |
| H | identitati NEVAZUTE la antrenare | **0.94** (sansa 0.25): generalizare zero-shot la cuvinte noi |
| I | categorii DESCOPERITE nesupervizat (clustering iterativ al deplasarilor) | clustere exacte: persoane / obiecte / verbe finite / participii / copule; decodare **0.96** |
| J1 | capacitate: 8/24/64 identitati x D=256/512/1024 | 8: .85/.95/.97; 24: .61/.74/.83; 64: .45/.56/.66 (sansa .016) |
| J2 | zgomot: 0..5 adverbe inserate aleator | 0.77 -> 0.70: robust |
| K | compozitie ierarhica "X spune ca <propozitie>" (3 forme ext. x 5 int. x 2 cadre, D=1024) | **0.88**; vorbitor 0.98, obiect 1.00, dest 0.86, agent 0.68 |

## Ce s-a invatat
1. Predictia (rapida sau lenta) prin memorie asociativa NU produce echivalenta: poate lega multe stari de acelasi raspuns.
   Echivalenta apare doar cand nivelul lent trebuie sa GENEREZE starea urmatoare printr-o operatie fixa.
2. Cuvantul trebuie factorizat: deplasarea e a categoriei (ce face), identitatea e in trasatura (ce este).
   Cu o singura deplasare per cuvant, identitatea cuvintelor anterioare "se scurge" in adresele celor urmatoare (imposibilitate algebrica).
3. Categoriile nu trebuie date: obiectivul generativ le produce singur (persoane 0.85 similaritate intre ele vs -0.03 cu obiectele),
   iar clusteringul iterativ le recupereaza exact.
4. Subiectul dinaintea verbului cere adresare amanata (uniunea de locatii din modelul Numenta / atribuirea rolurilor la verb).
5. Ierarhia merge cu acelasi mecanism: o ancora + drum relativ (zoom in), fara stiva explicita.

## Ce ramane nerezolvat / de facut la scalare
- Capacitatea e cea standard HDC: scade cu numarul de identitati la D fix; remediu cunoscut: D mare (10k+) si coduri rare pe blocuri.
- Agentul in formularile "de la A" / "de A" ramane cel mai slab (0.6-0.7); probabil coliziuni de adrese.
- "ca" ca deschizator de cadru si verbul ca declansator al bufferului sunt puse de mana; trebuie invatate (ex. din statistica: cuvintele dupa care predictia se "reseteaza").
- Semnalul de consecinta e dat de lumea sintetica; pe text real candidatul e propozitia/evenimentul urmator sau o modalitate pereche.
- Lipseste un control: un transformer mic pe aceeasi sarcina, cu acelasi numar de exemple.

## CONTROL (control_transformer.py): VSA vs transformer mic, aceeasi sarcina, aceleasi date
Sarcina: din propozitie prezice starea senzoriala a consecintei; decodarea rolurilor din PREDICTIE cu adresele lumii.
Ambele primesc aceeasi trasatura fixa per cuvant (calea "ce este" comuna). Antrenare pe formularile 0-3 si 8+8 identitati.
Transformer: d=128, 3 straturi, 4 capete, 854k parametri, 120k propozitii (3x mai multe). VSA: 10k parametri, 40k propozitii.

| test | VSA | transformer |
|---|---|---|
| identitati vazute, formulari vazute | 1.000 | 1.000 |
| identitati NEVAZUTE (4 noi; sansa .25) | **1.000** | 0.258 |
| formulare NEVAZUTA, identitati vazute (sansa .125) | 0.637 | 0.417 |
| formulare NEVAZUTA + identitati NEVAZUTE (sansa .25) | **0.742** | 0.240 |

Concluzie: la acest regim de date, transformerul memoreaza (perfect pe ce a vazut, la nivel de sansa pe identitati noi),
VSA-ul compune. Nota: corectia importanta descoperita aici: modelul si lumea trebuie sa imparta ACELEASI coduri de identitate
(in experimentele anterioare erau diferite, ceea ce limita calitatea predictiei, nu si evaluarile interne).
Limita controlului: un transformer mai mare cu mult mai multe date ar generaliza mai bine; aici conteaza eficienta datelor.

## PRIMUL TEST PE TEXT REAL (sick_test.py): SICK, 4927 perechi de test, scor uman de inrudire 1-5
Varianta G1 a modelului (fara adresare amanata, fara lume; semnal de echivalenta = perechi de antrenare cu scor >= 4).
| metoda | Pearson | Spearman |
|---|---|---|
| Jaccard (suprapunere de cuvinte) | 0.582 | 0.575 |
| sac de cuvinte (bundle de trasaturi aleatoare, fara deplasari) | 0.560 | 0.536 |
| deplasari aleatoare per cuvant | 0.223 | 0.197 |
| deplasari per cuvant, antrenate | 0.425 | 0.433 |
| deplasari pe categorii descoperite (clustering), antrenate | 0.448 | 0.453 |

REZULTAT NEGATIV: pe inrudire lexicala, ordinea/geometria strica fata de sacul de cuvinte. Contradictiile (negatii) NU sunt separate.
Ce a mers: clusteringul deplasarilor pe text real da clase semantice curate (man/person/woman/lady; boy/kid/child; a/the/one).
De ce nu e decisiv: (1) SICK rasplateste suprapunerea de cuvinte, nu rolurile (doar 29 perechi de test au aceleasi cuvinte in alta ordine);
(2) semnalul de antrenare e "inrudire", nu echivalenta de roluri; (3) lipsesc adresarea amanata si lumea; (4) ~11 min/antrenare pe acest CPU, deci subantrenat.
Testul potrivit pentru arhitectura: un set cu perechi care difera DOAR prin roluri/negatie, cu semnal de echivalenta dat de o lume (imagini: captions ale aceleiasi imagini).

## AL DOILEA TEST PE TEXT REAL, cu consecinta EXECUTABILA (wikisql_test.py): WikiSQL, 56k intrebari -> SQL
Lumea = interogarea SQL codata rol-umplutura (cod fix). Model = pasul uniform pe intrebare, obiectiv generativ. 2 epoci, CPU, ~70 s.
Decodare din PREDICTIE pe dev (3000): coloana selectata (dintre coloanele tabelului; sansa 0.15), agregator (6; majoritar 0.72), coloana conditiei 1.
| model | col. selectata | agregator | col. conditie | perechi acelasi schelet / diferit |
|---|---|---|---|---|
| sac de cuvinte (fara deplasari) | 0.19 | 0.17 | 0.15 | 0.77 / 0.43 |
| o singura deplasare pentru toate cuvintele (cod de pozitie) | **0.49** | 0.59 | 0.25 | 0.65 / 0.23 |
| deplasare per cuvant | 0.37 | **0.76** | 0.14 | 0.59 / 0.28 |
Citire: semnal real, dar modest. Modelul NU vede tabelul si nu are adresare amanata / categorii; cuvintele rare raman cu deplasari
neantrenate (de aceea "per cuvant" pierde pe coloana selectata si castiga pe agregator, care depinde de cateva cuvinte frecvente).
Urmatorul pas evident: deplasari pe categorii descoperite + adresare amanata + D mare; de asteptat un salt, nu o minune.

## CONTROL MLP (mlp_control.py): bundle fara ordine -> MLP (2.1M parametri) -> stare prezisa; aceleasi date si evaluare ca transformerul
| test | VSA | transformer | MLP pe bundle |
|---|---|---|---|
| identitati vazute, formulari vazute | 1.00 | 1.00 | 0.66 |
| identitati NEVAZUTE | 1.00 | 0.26 | 0.25 |
| formulare NEVAZUTA | 0.64 | 0.42 | 0.67 |
| ambele nevazute | 0.74 | 0.24 | 0.26 |
MLP-ul plafoneaza la 0.66 chiar pe date vazute: fara ordine nu poate distinge agentul de destinatar (ambii persoane in acelasi sac),
dar tocmai de aceea e indiferent la formulare (0.67 pe formularea nevazuta). Nici el nu generalizeaza la identitati noi.
Concluzie: ordinea e necesara (MLP), dar nu suficienta (transformer); VSA-ul e singurul care compune pe ambele axe la acest regim de date.

# RUNDA 2 (v2.py, v2b.py, v3.py, gates.py, gates_final.py, nest2.py): refacere pe noua intelegere
Schimbari de protocol: lumea = roluri explicite (R_rol (x) identitate), identitati IMPARTITE intre lume si model, decodare cu ADRESELE LUMII
(fara calibrare pe forma activa), fara T, 3 seed-uri, verbele = categorii individuale, P/O/COP = clase.
Model nou v2: adresare prin asteptari (cuvantul ia cea mai recenta asteptare neconsumata pentru categoria lui; altfel ramane in
buffer si se muta cand apare una); emitatorii = porti pe categorie; adrese relative pe (categorie emitator, categorie asteptata); origine fixa.
v3 (dual) = adresa v2 (+) drum pe categorii (grosier). Nicio regula pentru verb sau 'ca'.

## Tabel principal (3 seed-uri; antrenare T0-T3 cu 8+8 identitati) [vazute | identitati noi | formulare noua T4 | ambele]
| model | vazute | id. noi | T4 | ambele |
|---|---|---|---|---|
| v1 (drum + buffer declansat de verb, de mana) | 1.000 | 1.000 | 0.611±0.059 | 0.716±0.066 |
| drum pe categorii SINGUR (fara buffer, fara verb) | 1.000 | 1.000 | 0.546±0.083 | 0.620±0.069 |
| v2 asteptari, porti de mana | 1.000 | 1.000 | 0.181±0.056 | 0.401±0.135 |
| v2 asteptari aditive (out+in) | 1.000 | 1.000 | 0.127±0.082 | 0.242±0.096 |
| v2 toate portile deschise (lant bigram) | 0.787 | 0.834 | 0.121 | 0.269 |
| v2 toate inchise (totul la origine) | 0.129 | 0.223 | 0.136 | 0.239 |
| v3 dual (asteptari + drum) | 1.000 | 1.000 | 0.503±0.016 | 0.647±0.059 |
| v3 dual, porti INVATATE | 1.000 | — | 0.524±0.035 | — |

## CORECTIE MAJORA fata de runda 1
Drumul simplu pe categorii, FARA adresare amanata, rezolva perfect toate formularile vazute (inclusiv 'C primeste ...' si 'lui C A ii da B').
"Imposibilitatea algebrica" din runda 1 cerea egalitate exacta de adrese; cu decodare argmax, o adresa SUPRAPUSA (jumatate agent,
jumatate destinatar, pentru primul cuvant) e suficienta, pentru ca celalalt rol are un termen cu potrivire completa. Esecul din runda 1
(G1: T1 agent 0.00) venea din (a) evaluarea cu adresele formei active, care penalizeaza exact suprapunerea, si (b) trasaturi diferite
lume/model. Deci bufferul declansat de verb NU era necesar pentru formularile vazute. Costul suprapunerii se vede in MARJA:
  marja de decodare (corect - cel mai bun gresit): drum singur 0.154 | asteptari 0.496 | dual 0.480 ; predictie cos: 0.53 | 0.95 | 0.94
Asteptarile dau adrese curate (marja x3, predictie 0.95), drumul da robustete la constructii noi. Dualul le are pe amandoua.

## Declansatoarele POT fi invatate (gates.py): cautare discreta a portilor pe criteriul predictiei generative, fara etichete
Pornind cu toate inchise: deschise acceptate: '.', COP, are, catre, cumpara, da, de, ii, vinde, la. RESPINSE de fiecare data: P si O
(identitatile nu deschid cadre), plus dat/lui/nu/primeste/vandut (redundante). Portile invatate dau T0-T3 1.000 si T4 0.524 vs 0.464 cu
portile de mana (mai putine porti = mai putine tranzitii neantrenate). Criteriul a fost predictia, nu decodarea: structura "cine deschide
un cadru" a iesit din acelasi obiectiv ca restul.

## Ierarhie fara reguli (nest2.py): 'X spune ca <prop>', 3 forme ext. x 5 int. x 2 cadre, D=1024, fara regula pentru 'ca'
| model | vorbitor | agent | obiect | dest | T4 interioara nevazuta |
|---|---|---|---|---|---|
| drum singur (comutativ) | 1.00 | 1.00 | 1.00 | 0.99 | 0.80 (2 seed-uri) |
| dual (asteptari + drum) | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 (1 seed) |
Ancora si zoom-in-ul pus de mana in runda 1 (k.py) nu erau necesare: cu roluri explicite in lume, ambele variante rezolva imbricarea.

## Ce ramane adevarat din "scaparea principala" si ce nu
- Adevarat: inferenta miscarii din context lipsea; v2 o face prin asteptari si da adrese curate; portile (cine emite asteptari) se invata.
- Fals / corectat: nu era nevoie de nicio regula de mana (verb, 'ca') nici macar in v1; evaluarea era de vina.
- Nou si important: adresarea multiplicativa (suma de faze) e FRAGILA la o singura tranzitie neantrenata (o faza aleatoare strica toata
  adresa). Lantul de asteptari sufera cel mai tare (T4: 0.18); drumul aditiv pe categorii e mai robust (0.55); dualul mosteneste
  robustetea. Generalizarea la CONSTRUCTII noi e bariera reala a intregii familii, nu rolurile si nu identitatile.

# RUNDA 3 (v4.py): adrese REDUNDANTE pentru constructii noi — evaluare dupa PRINCIPII_EVALUARE.md
11 formulari x 2 cadre; antrenare pe 5 formulari (T0,1,2,3,6); test separat: formulari noi cu TOATE tranzitiile vazute (T9, T10) vs
formulari noi cu tranzitii/categorii nevazute (T4, T5, T7, T8). Adrese din lume, F comun (assert), 3 seed-uri, marja, sansa 0.125 / 0.25.
| cod de adresa | vazute | identitati noi | form. noi, tranzitii vazute | form. noi, tranzitii NEvazute |
|---|---|---|---|---|
| path (suma categoriilor din prefix) | 1.00 (m .25) | 1.00 (m .26) | 0.83 (m .11) | 0.60 (m .07) |
| chain (lant de asteptari) | 1.00 (m .51) | 1.00 (m .52) | 0.31 (m .04) | 0.51 (m .18) |
| bigram (categoria precedenta + proprie) | 1.00 (m .46) | 1.00 (m .47) | **1.00** (m .31) | 0.71 (m .16) |
| pos (pozitie absoluta) | 0.86 | 0.87 | 0.70 | 0.77 |
| path+chain | 1.00 (m .48) | 1.00 | 0.71 | 0.75 |
| path+bigram | 1.00 (m .45) | 1.00 | 0.98 (m .29) | 0.82 |
| path+chain+bigram | 1.00 (m .53) | 1.00 | 0.99 (m .19) | 0.82 |
| toate 4 | 1.00 (m .52) | 1.00 | 0.94 (m .19) | **0.90** (m .17) |
Concluzii:
- Ipoteza redundantei se CONFIRMA: pe tranzitii nevazute, de la 0.60 (path) / 0.51 (chain) la 0.90 (4 coduri); pe tranzitii vazute, de la 0.31 (chain) la 0.99.
- Surpriza: codul BIGRAM singur (adresa = categoria precedenta + proprie) e cel mai bun pe formulari noi cu tranzitii vazute (1.00, marja .31):
  rolurile sunt in mare parte LOCALE (prepozitia dinainte decide), iar primul cuvant ramane suprapus si e rezolvat de argmax.
- Nota: pe formulari noi cu tranzitii VAZUTE, "toate 4" (0.94) e SUB bigram singur (1.00): redundanta ajuta pe tranzitii noi
  (0.71 -> 0.90) si costa putin pe cele vazute (codurile mai fragile, chain/pos, adauga zgomot acolo unde bigram era deja suficient).
- Lantul de asteptari (v2) are marja maxima pe vazute dar e cel mai fragil pe nou; nu merita singur, merita in bundle.
- Toate codurile dau 1.00 pe identitati noi: compozitionalitatea pe identitati e o proprietate a familiei, nu a unui cod.
- Bariera ramane pe tranzitii/categorii cu adevarat noi (0.90, nu 1.00): acolo nu exista informatie in date; partea recuperata vine din
  codurile care nu depind de tranzitia noua (pos, path).

# RUNDA 4 (DGX, 2026-10-15): reproducere pe DGX si portare in torch/GPU (v4_torch.py) — fara schimbari de model
Aceeasi lume, aceleasi seed-uri, aceeasi evaluare ca runda 3. Loguri cu hash de commit in logs/.

## Reproducere (3 seed-uri; coloane: vazute | id. noi | form. noi tranz. vazute | form. noi tranz. NEvazute)
| rulare | toate 4 | observatie |
|---|---|---|
| v4.py numpy, CPU vechi (runda 3, log 2026-10-15_v4.txt) | 1.000 | 1.000 | 0.942±0.014 | 0.901±0.004 | referinta |
| v4.py numpy, CPU DGX (logs/..._1101_v4_d26b7db.txt) | 1.000 | 1.000 | 0.942±0.014 | 0.901±0.004 | IDENTIC la 3 decimale, toate 8 variantele |
| v4_torch.py GPU, batch=1 (logs/..._1107_v4_torch_c57d74a.txt) | 1.000 | 1.000 | 0.942±0.014 | 0.901±0.004 | IDENTIC pe toate 24 liniile (acuratete + marja) |
Concluzie: lumea/seed-urile/evaluarea sunt deterministe si portabile; portarea torch (autograd + torch Adam, float64) e echivalenta numeric
cu gradientul scris de mana la precizia raportata.

## Batch (v4_torch, 3 seed-uri, N=10000 propozitii, lr=0.02; batch>1 = mai putini pasi, acelasi numar de propozitii)
| batch | vazute | id. noi | form. noi tranz. vazute | form. noi tranz. NEvazute | s/varianta (antrenare) |
|---|---|---|---|---|---|
| 1 (= numpy) | 1.000 | 1.000 | 0.942±0.014 (m .19) | 0.901±0.004 (m .17) | 28.5 (encodare per pas) / ~20 (encodare o data) |
| 8 (logs/..._1120_v4_torch_fc97386.txt) | 1.000 | 1.000 | 0.976±0.015 (m .17) | 0.885±0.012 (m .15) | 4.0 (din care 1.0 encodare) |
| 32 (logs/..._1119_v4_torch_f8fedc5.txt) | 1.000 | 1.000 | 0.930±0.005 (m .19) | 0.887±0.016 (m .17) | 2.2 (din care 1.0 encodare) |
| 128 (seed 0 doar, informativ) | 1.00 | 1.00 | 0.89 | 0.87 | ~3 |
Pe celelalte coduri, batch 8/32 ramane in +-0.03 fata de batch 1, cu doua exceptii la batch 32: path 0.52 (vs 0.60) si path+chain 0.59/0.69
(vs 0.71/0.75) — codurile slabe pierd cand numarul de pasi scade de 32x la acelasi lr. Concluziile rundei 3 (ordinea codurilor, 0.90 pe
tranzitii nevazute cu toate 4) raman la batch 8-32; batch 128 incepe sa coste. Pentru lume mare: batch 8-32 si/sau mai multe propozitii.

## Timp (D=512, 10k propozitii, o varianta; 24 variante per rulare completa)
| | s/varianta | rulare completa |
|---|---|---|
| numpy, CPU vechi (runda 3) | ~77 | ~30 min |
| numpy, CPU DGX (20 nuclee, dar codul e secvential) | 24.8 | 10 min |
| torch GPU GB10, batch=1, encodare per pas | 28.5 | 11.5 min |
| torch GPU, batch=1, encodare o singura data | ~20 | — |
| torch GPU, batch=8 | 4.0 | 1.7 min |
| torch GPU, batch=32 | 2.2 | 1 min |
La D=512 si batch=1, GPU-ul NU castiga nimic: costul e Python (encodare, 10k pasi mici), nu aritmetica. Castigul (10x) vine din encodarea
o singura data + batch. Din cele 2.2 s la batch 32, 1 s e encodarea in Python (structura adreselor din v2/v4) — urmatorul loc de optimizat
daca lumea creste; GPU-ul devine decisiv abia la D >= 4096 si vocabular mare.

# RUNDA 5 (DGX, v5_scale.py, diag_v5.py): §5.1 — lume de 3x, 5 seed-uri, model v4 NESCHIMBAT
Lume: 6 cadre x 16 formulari (16 tipare comune, verbe diferite; fiecare forma verbala = categorie proprie), 18+18 identitati (12 antrenare),
imbricare 0-3 ('{X} spune ca <P>', '{X} zice ca <P>', '<P> , spune {X}', alese aleator pe nivel; vorbitorii X1 exterior..X3 = roluri in lume).
Antrenare: formulari T0,1,2,3,6,12,14; adancimi 0-2; 30k propozitii, batch 8, lr 0.02, D=1024. Log: logs/2026-10-15_1128_v5_scale_0defffd.txt.
Formulari noi cu TOATE tranzitiile vazute: T9, T10; cu tranzitii/categorii NEvazute: T4, T5, T7, T8, T11, T13, T15. Sansa 1/12 = 0.083 (id. noi 1/6).

| cod | vazute | id. noi | form. noi, tranz. vazute | tranz. NEvazute | imbricare ad.1-2 (vazuta) | ad.3 (NEvazuta) |
|---|---|---|---|---|---|---|
| path | 0.950 (m .23) | 0.952 | 0.829±0.017 | 0.575±0.006 | 0.832 | 0.581 |
| chain | 1.000 (m .48) | 1.000 | 0.232±0.018 | 0.386±0.010 | 0.925 | 0.661 |
| bigram | 0.956 (m .47) | 0.984 | 0.918±0.036 | 0.623±0.022 | 0.907 | 0.651 |
| pos | 0.889 | 0.895 | 0.695 | 0.705 | 0.805 | 0.555 |
| path+bigram | 0.993 | 1.000 | 0.994±0.005 | 0.816±0.009 | 0.920 | 0.652 |
| path+chain+bigram | 1.000 (m .51) | 1.000 | **0.998±0.003** (m .23) | 0.821±0.006 (m .16) | 0.938 | 0.673 |
| toate 4 | 1.000 (m .51) | 1.000 | **0.841±0.002** (m .24) | **0.984±0.006** (m .19) | 0.943 | 0.687 |

## Criteriul §5.1 (medii in +-0.03 fata de runda 3)
- path+chain+bigram: 0.998 / 0.821 (lume mare) vs 0.987 / 0.824 (lume mica) => REPRODUS, stabil intre lumi.
- toate 4: 0.841 / 0.984 vs 0.942 / 0.901 => NU se reproduce; cele doua coloane se INVERSEAZA. Cauza (diag_v5.py, mai jos): codul pos.
- Identitati noi 1.00 pentru orice bundle: confirmat si pe lume mare. path singur cade la 0.95 pe vazute (mai multe categorii la D=1024).

## Diagnostic 1 (diag_v5.py, seed 0, per formulare x rol): de ce 'toate 4' pierde pe T10 si castiga pe tranzitii nevazute
T10 '{A} ii V {B} lui {C}': pos singur A 0.00 (toate 240 confuziile A->C); toate 4: A 0.15 (A->C); path+chain+bigram: A 0.96. T9: toti 1.00.
Mecanism: in antrenare, singurul tipar cu o persoana la pozitia 5 e T1 '{C} VR {B} de la {A}' (acolo e AGENTUL). pos invata "persoana de la
pozitia 5 = A". In T10 la pozitia 5 e C: pos voteaza cu potrivire COMPLETA ca C e agentul; primul cuvant (A) are in TOATE codurile adresa
suprapusa A/C (T0 vs T1 incep amandoua cu o persoana), deci jumatate de vot. Un vot complet gresit bate patru jumatati corecte.
Pe tranzitii NEvazute (per formulare, A/B/C): path+chain+bigram esueaza exact unde apare o CATEGORIE noua sau o tranzitie noua la final:
  T4 'de catre {A}' A 0.07 ('catre' nou) | T11 '{A} il V' A 0.06 ('il' nou) | T8 '{A} V lui {C} {B}' B 0.32 (P->O nou) | T13 '... {A} {B}' B 0.18 (P->O nou);
  toate 4 le ridica la 0.72 / 0.93 / 1.00 / 1.00 prin pos.
Concluzie: pos = PRIOR DE POZITIE ABSOLUTA. Ajuta cand celelalte coduri au zero informatie (categorie noua) si strica cand regularitatea
de pozitie din antrenare e falsa (T10). Ambele efecte sunt proprietati ale SETULUI DE FORMULARI, nu ale mecanismului. Cifra de raportat ca
"modelul" e path+chain+bigram; pos se raporteaza separat, ca prior.

## Diagnostic 2: imbricarea. Adancime 1 = 1.00; adancime 2 (VAZUTA la antrenare) ~0.85; adancime 3 NEvazuta
Per rol, seed 0, T0, path+chain+bigram: ad.1: A 1.00 B 1.00 C 1.00 X1 1.00 | ad.2: A 0.84 B 1.00 C 1.00 X1 0.80 X2 0.45 (X2->A 175/240, A->X2, X1->X2)
  | ad.3: A 0.59 B 1.00 C 1.00 X1 0.74 X2 0.30 X3 0.09. Niciun cod singur nu separa X2: path 0.28, chain 0.38, bigram 0.33, toate 4 0.53.
Cauza (logica, confirmata empiric): '{X1} spune ca {A} V ...' (ad.1) si '{X1} spune ca {X2} zice ca {A} V ...' (ad.2) au PREFIX IDENTIC
pana la al patrulea cuvant ('P spune ca P'). Orice adresa calculata din prefix (path, chain, bigram, pos, orice bundle) e identica pentru
A la ad.1 si X2 la ad.2, iar lumea cere R_A, respectiv R_X2 => adresa suprapusa => ~0.5. Rolul e decis de cuvantul URMATOR ('zice' vs verb).
Asta e, in forma curata, problema "adresarii amanate" din runda 1, retrasa in runda 2: pentru adancime 1 si cadre simple NU e necesara
(confirmat din nou: ad.1 = 1.00 fara reguli); pentru imbricare >= 2 informatia NU e in prefix, deci e necesara o forma de amanare /
reasezare retroactiva / privire inainte. Niciun cod cauzal nu o poate rezolva, indiferent de D sau de date.
Adancime 3: X3 nu a fost NICIODATA tinta la antrenare (adancimea maxima de antrenare e 2) => la sansa PRIN CONSTRUCTIE (0.08-0.09 ~ 1/12);
cifra "ad.3" din tabel amesteca aceasta imposibilitate cu degradarea reala a lui A, X1, X2 (0.59 / 0.74 / 0.30). O adancime noua = un rol
nou; se raporteaza separat (regula noua in PRINCIPII_EVALUARE).

## Ce ramane / ce se schimba fata de §4
- RAMANE SIGUR (acum 5 seed-uri, lume de 3x): factorizare + obiectiv generativ => roluri aliniate; identitati noi 1.00; path+chain+bigram
  0.99 / 0.82 stabil intre lumi; bariera = tranzitii/categorii cu adevarat noi (acolo doar un prior de pozitie mai ajuta).
- SE CORECTEAZA: "redundanta 0.60 -> 0.90" devine "0.58 -> 0.82 (coduri structurale) + prior de pozitie dependent de date";
  "imbricare 1.00 fara reguli" devine "adancime 1 = 1.00; adancime >= 2 NU e rezolvabila cu adrese din prefix".
- NOU: prima dovada curata ca o forma de adresare amanata E necesara (nu pentru verb/cadre, ci pentru prefixe identice cu roluri diferite).
Timp: 21.7 s/varianta (30k propozitii, D=1024, batch 8), din care ~13 s encodare Python; rulare completa (7 variante x 5 seed-uri) 12.7 min.
