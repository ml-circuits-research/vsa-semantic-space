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
