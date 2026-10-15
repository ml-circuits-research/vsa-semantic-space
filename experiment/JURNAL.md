# Jurnal de lucru (o intrare per sesiune; format in GIT_WORKFLOW.md §3)

## 2026-10-14/15 — sesiunea initiala (Claude + autor), CPU, numpy
Rulat: sim_deplasari.py, camp.py, k.py, control_transformer.py, mlp_control.py, sick_test.py, wikisql_test.py, v2.py, v3.py, gates.py, nest2.py, v4.py
Rezultat: vezi REZULTATE.md (3 runde). Pe scurt: factorizare + obiectiv generativ => roluri aliniate; porti invatabile; redundanta 0.60 -> 0.90.
Surpriza / problema: doua greseli de evaluare in runda 1 (adrese din forma activa; F diferit lume/model) au produs "reguli necesare" false.
Schimbat in cod: evaluarea refacuta (v2.evaluate: adrese din lume); PRINCIPII_EVALUARE.md scris.
Urmatorul pas: portare torch; v4 cu 5 seed-uri; WikiSQL cu modelul complet + baseline BoW+LR.
