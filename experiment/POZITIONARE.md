# Pozitionare, prezentarea arhitecturii, controale, credibilitate

## Numele arhitecturii si fraza de un rand
Propunere de nume: "Path-Integrated Semantic State" (PISS e nefericit; alternativ "Displacement Semantics" sau "Role-Emergent VSA" / REVSA).
Fraza: *Un cuvant este o pereche (identitate fixa, deplasare invatata pe categorie); starea unei propozitii este bundle-ul identitatilor
legate la locatiile atinse prin integrarea deplasarilor; antrenarea sa prezica generativ, printr-o operatie fixa, starea consecintei face ca
formulari diferite sa converga la aceeasi stare, rol cu rol, iar categoriile sintactice sa emearga fara supervizare.*

## Cum se prezinta arhitectura (figura 1)
Trei panouri, nu unul:
 (a) pasul uniform: L <- L * m_cat(w); S <- S + L * f_w, cu o propozitie activa si una pasiva desenate ca doua drumuri pe o grila care
     ating aceleasi (adresa, identitate) in alta ordine;
 (b) coduri de adresa REDUNDANTE (path, bigram, chain, pos) in bundle: o tranzitie de categorii neantrenata nu randomizeaza toata adresa
     (runda 3: 0.60 -> 0.90 pe constructii noi). [RETRAS: "adresarea amanata" (buffer + zoom-in la "ca") din runda 1; infirmata in runda 2.]
 (c) obiectivul: S * T ≈ S_lume, cu T fix, versus memoria asociativa (care NU aliniaza). Acest panou e cheia argumentului.
Tabelul 1 = rezultatele negative (A, C, D, E2) cu explicatia algebrica a fiecaruia. E neobisnuit, dar e ce da credibilitate:
arata ca ingredientele nu sunt alese arbitrar, ci fortate.

## Trei articole posibile, de la cel mai sigur la cel mai ambitios
1. **Nota de principiu** (workshop neuro-simbolic la NeurIPS/ICLR, sau Neural Computation / Frontiers in Comp. Neurosci.):
   "Necessary conditions for role emergence in vector-symbolic sequence models". Contributii: (i) adrese suprapuse + decodare argmax:
   alinierea rolurilor nu cere egalitate exacta de adrese [RETRAS: "imposibilitatea algebrica a deplasarii per cuvant" — corecta doar pentru
   egalitate exacta, irelevanta cu decodare argmax; vezi runda 2]; (ii) asociativ vs generativ; (iii) emergenta categoriilor;
   (iv) portile (cine deschide un cadru) se invata prin cautare pe criteriul predictiei; (v) generalizare zero-shot la identitati noi vs
   transformer mic.
   Cerinte minime: 5 seed-uri, F comune, lume cu 4-6 cadre si 8+ formulari, controlul word2vec pentru categorii, controlul "memorie
   asociativa cu gat de sticla". Realizabil in 2-3 saptamani pe DGX. Credibilitate: buna, daca pretentiile raman la nivelul sintetic.
2. **Articol empiric** (ACL/EMNLP findings, sau CoNLL): text->SQL cu stare citibila. Contributii: eficienta datelor (curbe 1k/5k/56k),
   interpretabilitate rol cu rol, vocabular nou fara reantrenare. Controale OBLIGATORII: BoW + regresie logistica; transformer mic
   reglat; un BERT-small fine-tuned; toate la aceleasi date. Realizabil in 1-2 luni. Credibilitate: depinde integral de controale.
3. **Articol de teorie/neuro** (legatura cu TBT / celule grila / predictive coding): doar daca declansatoarele se invata singure
   (experimentul 3 din README_DGX). Fara asta, legatura cu cortexul ramane in sectiunea de discutie a articolului 1, nu in titlu.

## Unde e bariera, cinstit
- Pe text fara lume (SICK) am pierdut fata de sacul de cuvinte. Nu publica nimic pe text general inainte de a avea un set unde echivalenta
  e data de executie/imagine si unde negativele grele (roluri inversate) sunt masurate separat.
- [RETRAS dupa runda 2] Declansatoarele (verb -> flush, "ca" -> cadru) puse de mana nu erau necesare: drumul pe categorii rezolva
  formularile vazute si imbricarea fara ele; portile (cine emite asteptari) se invata prin cautare (gates.py). Ramane de aratat ca se invata
  si prin gradient, pe >=3 lumi (00_CITESTE_INTAI.md §5.2).
- Capacitatea: 64 identitati -> 0.66 la D=1024. Standard pentru HDC; se rezolva cu D mare si coduri rare, dar trebuie aratat.
- Controlul cu transformer e slab (netunat). Pana nu e tunat, formularea permisa e "la acest regim de date, un transformer mic antrenat
  identic nu generalizeaza", nu "transformerele nu pot".

## Ce comparatii sunt facute / planificate
Facute: transformer mic (numpy, 854k param., gradient-check) pe sintetic; sac de cuvinte si Jaccard pe SICK; sac de cuvinte si cod de pozitie
pe WikiSQL. Planificate (README_DGX): BoW+LR, transformer tunat, BERT-small, word2vec pentru categorii, memorie asociativa cu gat de sticla,
curbe de date, 5 seed-uri. Un MLP simplu (bundle -> MLP -> stare) merita adaugat ca baseline "fara ordine, cu capacitate".

## Pe ce sa NU se bazeze credibilitatea
Nu pe analogia cu creierul, nu pe numarul mic de parametri in sine (HDC-ul e mereu mic), nu pe un singur benchmark. Pe: rezultate negative
explicate, ablatii (fiecare ingredient scos pe rand: factorizare, amanare, operatie fixa), si controale la aceleasi date.

## Lucrari de citat (verifica fiecare inainte de a cita; lista e din memorie)
Plate 1995/2003 (HRR); Kanerva 2009 (HDC); Gayler 2003 (VSA); Smolensky 1990 (tensor product); Eliasmith 2012 (SPA);
Frady, Kleyko, Sommer 2018-2022 (fractional power encoding; resonator networks; celule grila ca FPE);
Kleyko et al. 2022 (survey HDC/VSA); Hawkins et al. 2017/2019, Lewis et al. 2019 (Thousand Brains, displacement cells);
Rao & Ballard 1999, Friston 2010 (predictive coding); Elman 1990 (RNN ca traiectorie); Kamp 1981, Heim 1982, Veltman 1996 (semantica dinamica);
Fillmore 1982 (frame semantics); Hasson et al. 2008 (temporal receptive windows); Katharopoulos et al. 2020 (atentie liniara = bundle de bind-uri);
Schlag & Schmidhuber 2018 (fast weights, TPR); Lake & Baroni 2018 (SCAN, generalizare compozitionala); Zhong et al. 2017 (WikiSQL); Marelli et al. 2014 (SICK).
