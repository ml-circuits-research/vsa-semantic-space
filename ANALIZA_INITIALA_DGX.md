# Analiza `experiment/` — ce e, ce e stabilit, ce e neconfirmat, discrepante

## Ce e proiectul

Ipoteza de pornire: a intelege o fraza inseamna a parcurge un drum intr-un spatiu, iar doua fraze cu acelasi
sens sunt drumuri diferite care ajung in acelasi punct. Spatiul e construit cu VSA/FPE (vectori complecsi de
modul 1, D = 512-1024). Trei operatii:
- bind: o deplasare (inmultire element cu element = adunare de faze);
- bundle: o insumare care topeste termenii intr-unul singur (singura operatie ireversibila);
- cleanup: alegerea celui mai apropiat concept cunoscut (argmax pe cosinus).

Modelul curent (v4.py): fiecare cuvant are o identitate fixa, aleatoare, plus o adresa calculata din context.
Adresa e un bundle din 4 coduri invatate pe CATEGORII, nu pe cuvinte:
- path:   suma deplasarilor categoriilor din prefix;
- bigram: categoria precedenta + cea proprie;
- chain:  lantul de asteptari;
- pos:    pozitia absoluta (x) categoria.

Starea frazei: S = sum adresa (x) f_w. Obiectiv unic, generativ: cos(S, S_lume), unde lumea e data de roluri
explicite si foloseste ACELEASI identitati ca modelul. Evaluarea decodeaza rolurile cu adresele lumii si
raporteaza acuratete + marja.

## Ce e stabilit (3 runde, numpy, CPU)

- Factorizarea cuvantului (identitate + deplasare pe categorie) si obiectivul generativ sunt necesare.
  Predictia asociativa si varianta cu ambele parti invatate colapseaza.
- Pe formularile vazute rolurile se aliniaza perfect. Pe identitati noi iese 1.00, dar asta decurge din
  constructie (identitatea nu intra in parametri).
- Imbricarea "X spune ca ..." merge FARA nicio regula de mana. Corectia majora din runda 2: regulile
  "verbul goleste bufferul" si "zoom-in la 'ca'" din runda 1 veneau din doua erori de evaluare
  (adrese din forma activa; F diferit lume/model), nu erau necesare.
- Portile (care categorii deschid un cadru) se pot invata prin cautare discreta: verbele si prepozitiile
  deschid, persoanele si obiectele nu.
- Bariera reala sunt CONSTRUCTIILE noi (tranzitii de categorii nevazute). Bundle-ul redundant de coduri
  urca acuratetea pe ele de la 0.60 la 0.90.

## Ce e inca neconfirmat

- Imbricarea pe dual are 1 seed; "drum singur" are 2. Categoriile pe SICK si portile invatate au fost vazute
  pe o singura lume.
- Lipsesc controalele cheie: word2vec pentru categorii, memorie asociativa cu gat de sticla, transformer
  TUNAT (cel actual e netunat, 854k param).
- WikiSQL: 0.49 o singura data, fara baseline BoW+LR. SICK: negativ (0.45 vs 0.575 Jaccard).
- Detaliu din v4: pe formulari noi cu tranzitii VAZUTE, "toate 4" da 0.94, sub bigram singur (1.00).
  Redundanta ajuta unde tranzitiile sunt noi si costa ceva unde sunt vazute. De notat in REZULTATE.md.

## Discrepante in documente

- POZITIONARE.md inca propune "imposibilitatea algebrica" ca prima contributie a articolului 1 si
  "adresarea amanata" (buffer + zoom-in) in panoul (b) al figurii 1. Ambele au fost infirmate in runda 2.
- README_DGX.md amesteca planul vechi (de dupa runda 1: "adresare amanata cu declansator invatat", k.py
  zoom-in pe Spider) cu cel de dupa runda 2 (sectiunea de la final). Primele sectiuni sunt depasite.
- Logurile din logs/ nu au formatul run.sh (lipseste hash-ul de commit din nume): au fost produse inainte
  de a exista run.sh.
- README_RUN.md numeste camp.py "MODELUL FINAL" si listeaza "adresare amanata, zoom-in" ca parti ale lui;
  00_CITESTE_INTAI.md spune ca modelul curent e v4.py. 00_CITESTE_INTAI are prioritate.

## Mediul de lucru

- Pe masina NU e instalat nici numpy, nici torch. Python 3.12.3. GPU: NVIDIA GB10 (DGX Spark, aarch64).
- Trebuie un venv cu numpy si torch cu CUDA pentru aarch64.

## Ce e de facut (planul din 00_CITESTE_INTAI.md §5-6)

1. Mediul: venv + numpy + torch (CUDA aarch64).
2. Ziua 1: portare v4.py in torch (faze ca nn.Parameter, torch.polar, autograd in loc de gradient manual,
   batch pe lungime). Verific ca seed 0 reproduce tabelul rundei 3 in +-0.03. Apoi 5 seed-uri pe lume de
   2-3x mai mare (4-6 cadre, 10-20 formulari, 3-4 nivele de imbricare).
3. Ziua 2: porti invatate prin gradient (sigmoid + amestec moale), pe >=3 lumi. Apoi controlul word2vec si
   memoria asociativa cu gat de sticla.
4. Mai departe: transformer tunat + curbe de date 1k/5k/50k; capacitate (D 4096-16384, coduri rare pe
   blocuri); WikiSQL cu v4 complet + baseline BoW+LR; apoi Spider.
5. Disciplina: commit inainte de fiecare rulare -> ./run.sh -> intrare in JURNAL.md -> tabel in REZULTATE.md.
   Evaluarea respecta checklistul din PRINCIPII_EVALUARE.md.

Propunere de start: pasii 1-2 (mediu, portare v4, reproducere seed 0). Corecturile la POZITIONARE.md si
README_DGX.md separat, intr-un commit mic.
