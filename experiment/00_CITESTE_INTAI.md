# Spatiu semantic ca drum: istoria proiectului, stare, perspective
(document de predare pentru o sesiune noua; citeste-l integral inainte de orice rulare)

## 0. Ideea de la care s-a pornit (formularea originala a autorului)
Interpretarea textului = traiectorie intr-un spatiu (graf / hipergraf / latice). Contextul e un nod; fiecare cuvant "dezvolta" spatiul dupa
reguli codate in structura lui; fraze diferite cu acelasi sens sunt drumuri diferite spre acelasi punct. Intrebari: se poate antrena un
sistem sa invete ponderile cuvintelor ca sa obtina echivalentele? Exista legatura cu "o mie de creiere" (Hawkins), cu VSA/HDC, cu fibrati?

## 1. Cadrul matematic ales
Vector Symbolic Architecture cu Fractional Power Encoding: vectori complecsi de modul 1 (faze), D = 512-1024.
  bind = inmultire element cu element (= adunare de faze) = DEPLASARE rigida (muchie etichetata, izometrie)
  bundle = adunare de vectori = TOPIRE intr-un corp comun (centru de masa, hipermuchie); singura operatie ireversibila
  cleanup = argmax cosinus pe candidati cunoscuti = "gravitatie" spre cel mai apropiat concept
Lectura geometrica: spatiul e un tor D-dimensional; bind-ul cu un rol te muta intr-o "fibra" (copie a spatiului); fibrati de fibrati =
bind-uri inlantuite; "fractal algebric" = bind contractiv (IFS) — discutat, NU implementat (vezi 7).

## 2. Modelul care a ramas in picioare (dupa 3 runde)
Un cuvant = pereche (ce este, ce face):
  - identitate f_w: vector fix, aleator, niciodata invatat (una per cuvant)
  - adresa: calculata din CONTEXT prin coduri invatate pe CATEGORII (nu pe cuvinte): persoane, obiecte, copule = clase; verbe si
    cuvinte functionale = categorii individuale
Pas unic, identic pentru orice cuvant:  S <- S + adresa(context) (x) f_w
Starea unei propozitii = S (bundle). Doua formulari sunt echivalente daca au depus aceleasi identitati la aceleasi adrese.
Adresa = BUNDLE DE MAI MULTE CODURI (runda 3, v4.py):
  path   = suma deplasarilor categoriilor din prefix (aditiv, comutativ, robust)
  bigram = categoria precedenta + proprie (local; surprinzator de puternic)
  chain  = lantul de asteptari: un cuvant ia cea mai recenta asteptare neconsumata pentru categoria lui; emitatorii (porti pe
           categorie) emit asteptari relative la adresa lor (precis, marja mare, fragil pe nou)
  pos    = pozitie absoluta (x) categorie
Lumea: roluri explicite S_lume = sum_rol R_rol (x) f_umplutura, cu ACELEASI f ca modelul.
Obiectiv unic: maximizeaza cos(S, S_lume). Generativ, fara etichete de parafraze, fara negative. Adam, lr 0.02, 10-15k propozitii.
Evaluare: decodeaza rolul r din S cu adresa LUMII R_r: argmax_x cos(S (x) conj(R_r), f_x). Raporteaza acuratete + marja.

## 3. Istoria, pe runde, cu greselile
RUNDA 1 (sim_deplasari.py, f.py, camp.py, k.py, control_transformer.py, mlp_control.py)
  - A: predictie cuvant->cuvant: colaps (totul similar). C: predictia consecintei prin memorie asociativa: prezice 100% dar NU aliniaza
    starile. D: prototipuri: memoreaza. E: generativ cu ambele parti invatate: colaps. => predictia produce echivalenta doar daca e
    GENERATIVA printr-o operatie fixa si lumea e fixa. (CONFIRMAT si in rundele 2-3.)
  - F: factorizarea cuvantului (deplasare pe categorie, identitate fixa) a fost necesara. (CONFIRMAT.)
  - G/K: "adresare amanata" (bufferul dinaintea verbului) si "zoom-in" la 'ca' puse de mana. INFIRMATE in runda 2: nu erau necesare;
    aparentul esec venea din (a) evaluare cu adrese din forma activa (penaliza adrese suprapuse legitime) si (b) BUG: lume si model cu
    identitati diferite. Toate cifrele rundei 1 pe formulari neactive sunt subestimari.
  - "Imposibilitatea algebrica a deplasarii per cuvant": argument corect pentru egalitate EXACTA de adrese, irelevant cu decodare argmax.
  - H: identitati noi zero-shot 0.94: CONFIRMAT (1.00 in rundele 2-3), dar e proprietate de constructie (identitatea nu intra in parametri)
    si presupune categoria cunoscuta.
  - I: categorii descoperite nesupervizat prin clustering al deplasarilor: clustere exacte. CONFIRMAT pe sintetic; pe text real (SICK) da
    clase curate (man/person/woman; boy/kid/child). Lipseste controlul word2vec.
  - Control transformer (854k param, netunat) si MLP pe sac: ambele 0.25 pe identitati noi (sansa) vs VSA 1.00. Valabil doar ca
    "la acest regim de date"; transformerul nu a fost tunat.
  - SICK (text real, inrudire 1-5): NEGATIV (Spearman 0.45 vs 0.575 Jaccard). SICK nu masoara roluri; nu infirma teza, dar arata ca pe text
    fara lume semnalul e zgomotos. WikiSQL (text->SQL, consecinta executabila): semnal modest (col. selectata 0.49 vs sansa 0.15, fara tabel).
RUNDA 2 (v2.py, v3.py, gates.py, nest2.py)
  - Drumul simplu pe categorii, FARA reguli, rezolva perfect toate formularile vazute si imbricarea 'X spune ca <prop>' (corectia majora).
  - v2 (asteptari): marja x3 (0.50 vs 0.15), predictie 0.95 vs 0.53, dar fragil pe formulari noi (0.18).
  - PORTILE SE INVATA: cautare discreta pe criteriul predictiei deschide verbe/prepozitii/copula/punct si RESPINGE persoanele si obiectele.
    Portile invatate > portile de mana (0.52 vs 0.46 pe formulare noua).
RUNDA 3 (v4.py, PRINCIPII_EVALUARE.md)
  - Bariera reala = CONSTRUCTII noi (tranzitii de categorii nevazute): o faza neantrenata randomizeaza o adresa multiplicativa.
  - Redundanta (bundle de coduri) o ridica de la 0.60 la 0.90; bigram singur = 1.00 pe formulari noi din tranzitii vazute.

## 4. Ce indica experimentele (cu nivelul de incredere)
  SIGUR (3 seed-uri, evaluare corecta): factorizare necesara; obiectiv generativ necesar; roluri aliniate pe 11 formulari x 2 cadre;
    identitati noi 1.00; imbricare 1.00 fara reguli; redundanta ajuta la constructii noi; portile se pot invata prin cautare.
  PROBABIL (1-2 seed-uri sau un singur set): categorii emergente pe text real; porti invatate = acelasi set pe alte lumi.
  SLAB / DE VALIDAT: avantajul fata de transformer (control netunat); WikiSQL 0.49; orice afirmatie despre text general.
  NEGATIV: SICK; v2 singur pe constructii noi; predictia asociativa pentru echivalenta.

## 5. Ce trebuie VALIDAT la scalare (in ordinea importantei)
  1. Reproducere cu 5 seed-uri a tabelelor din RUNDA 2-3 (v2.py, v4.py), lume de 2-3x mai mare (4-6 cadre, 10-20 formulari, 3-4 nivele de
     imbricare). Criteriu: medii in +-0.03 fata de cele de aici.
  2. Porti invatate PRIN GRADIENT (sigmoid + amestec moale adresa provizorie/asteptata). Criteriu: converg la "identitatile inchise,
     actiunile/relatiile deschise" pe >=3 lumi diferite. Daca da => "totul emerge dintr-un singur obiectiv" e sustinut.
  3. Controlul word2vec pentru categorii (aceleasi date): daca da clustere la fel de curate, emergenta categoriilor nu e meritul arhitecturii.
  4. Control "memorie asociativa cu gat de sticla": daca aliniaza la fel de bine, "generativ necesar" se restrange la "capacitate limitata necesara".
  5. Transformer TUNAT (+ dropout, weight decay, lr sweep) si BERT mic, la 1k/5k/50k exemple: curba de eficienta a datelor, nu un punct.
  6. Capacitate: 64-1000 identitati la D = 4096-16384; coduri rare pe blocuri; cleanup la vocabular mare.
  7. Text real cu consecinta executabila: WikiSQL cu v4 (toate 4 codurile) + porti invatate + categorii descoperite + tabelul vizibil.
     Baseline obligatoriu: sac de cuvinte + regresie logistica. Apoi Spider (SQL imbricat).
  8. Semnal de consecinta fara lume sintetica: propozitia urmatoare / legende multiple per imagine (COCO) / traduceri (OPUS) cu negative grele.

## 6. Pasi concreti pe DGX (torch)
  Ziua 1: portare (fazele ca nn.Parameter; torch.polar; cumsum pe categorii; autograd inlocuieste grad_cos; batch pe lungime);
          verifica pe v4.py ca reproduce tabelul RUNDA 3 cu seed 0; apoi 5 seed-uri, lume mare.
  Ziua 2: porti prin gradient; word2vec control; memorie asociativa cu gat de sticla.
  Ziua 3-5: transformer tunat + curbe de date; capacitate.
  Saptamana 2: WikiSQL/Spider cu v4; baseline BoW+LR.
  Daca 2 si 3 ies bine: scrie nota de principiu (POZITIONARE.md, articolul 1). Daca 7 iese bine: articolul empiric.

## 7. Perspective si idei neimplementate
  - Bind contractiv (IFS, fractal algebric): adrese cu scara, adancime citibila, zgomot neacumulat; cost: inversare fragila.
  - Spatiu de adrese necomutativ (permutari) pentru ordine nativa; comutativ doar in interiorul unui cadru.
  - Asteptari cu identitate specifica (nu doar categorie) = predictie lexicala top-down; ar reduce suprapunerea primului cuvant.
  - Multirezolutie prin ceasuri (nivel rapid pe cuvinte, lent pe propozitii) in loc de ancore.
  - Legatura cu TBT/celule grila: motivatie si interpretare, NU rezultat; ramane in discutie, nu in titlu.

## 8. Principii de evaluare (rezumat; integral in PRINCIPII_EVALUARE.md)
  adrese din lume; F comun (assert); sansa tiparita; 3 axe vazut/nevazut separate (identitati / formulari cu tranzitii vazute /
  tranzitii nevazute); marja raportata; >=3 seed-uri; control trivial in acelasi tabel; nicio calibrare din test; traseaza un exemplu
  inainte de antrenare; un esec se verifica intai ca evaluare, apoi date, apoi optimizare, abia apoi arhitectura.

## 9. Harta fisierelor
  GIT_WORKFLOW.md (cum comitem si comunicam prin GitHub) | run.sh (ruleaza + salveaza log cu data si hash) | JURNAL.md | logs/
  00_CITESTE_INTAI.md (acesta) | README_RUN.md (cum rulezi, ce fisier ce face) | README_DGX.md (plan, surse de date, portare)
  REZULTATE.md (toate tabelele, 3 runde) | INTERPRETARE.md (cum se citeste fiecare metrica) | POZITIONARE.md (articole, controale, citari)
  PRINCIPII_EVALUARE.md | cod: camp.py (lumea + v1), v2.py (asteptari, RoleWorld, evaluate), v4.py (coduri redundante; MODELUL CURENT),
  gates.py (cautarea portilor), nest2.py (imbricare), control_transformer.py, mlp_control.py, sick_test.py, wikisql_test.py | log_*.txt
  Comanda de verificare rapida (2 min pe CPU): python3 v2.py ; comanda principala (25 min CPU, secunde pe GPU): python3 v4.py
