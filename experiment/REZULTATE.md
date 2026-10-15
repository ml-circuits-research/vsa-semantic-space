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
