# Cum lucram prin GitHub (ca o sesiune noua de Claude sa preia contextul fara pierderi)

## 1. Initializare (o singura data, pe DGX)
    unzip spatiu_semantic_vsa.zip -d spatiu-semantic && cd spatiu-semantic
    git init && git add -A && git commit -m "import: starea de la 2026-10-15 (3 runde sintetic, SICK, WikiSQL)"
    # repo pe GitHub (privat e ok), apoi:
    git remote add origin git@github.com:<user>/spatiu-semantic.git && git push -u origin main

## 2. Ce se comite (TOT ce e relevant; lipsa rezultatelor e singura pierdere reala)
  - cod: *.py (orice schimbare, oricat de mica)
  - rezultate brute: logs/<data>_<script>_<hash>.txt  (produse automat de run.sh, vezi 4)
  - REZULTATE.md: tabelele noi, in formatul existent (model | metrica | seed-uri); JURNAL.md: o intrare per sesiune (vezi 3)
  - PRINCIPII_EVALUARE.md daca se schimba evaluarea (si re-ruleaza un experiment vechi cu noua evaluare, ca punct de legatura)
  - date mici (< 50 MB) in corp/; date mari NU (pune in corp/README.md comanda de descarcare)
  - NU: checkpoint-uri mari, __pycache__, fisiere temporare (.gitignore inclus)

## 3. JURNAL.md: o intrare per sesiune de lucru, 3-8 randuri, format fix
    ## 2026-10-16 (seara) — <titlu scurt>
    Rulat: run.sh v4 (5 seed-uri, D=4096)           -> logs/2026-10-16_v4_a1b2c3d.txt
    Rezultat: form. noi tranz. nevazute 0.91±0.02 (confirma 0.90 de pe CPU)
    Surpriza / problema: ...
    Schimbat in cod: v4.py: ... (de ce: ...)
    Urmatorul pas: ...
  Scrie si cand NU a mers. Un esec notat valoreaza cat o reusita.

## 4. run.sh: ruleaza un script si salveaza logul cu data + hash de commit in nume (nu depinde de disciplina)
    ./run.sh v4.py            # -> logs/2026-10-16_v4_a1b2c3d.txt  (+ tipareste si pe ecran)
    ./run.sh gates.py
  run.sh refuza sa ruleze daca exista modificari necomise (ca logul sa corespunda EXACT unui commit). Comite intai, apoi ruleaza.

## 5. Mesaje de commit: spun DE CE, nu CE
    bine:  "evaluare: adrese din lume in loc de forma activa (penaliza adrese suprapuse)"
           "v4: cod bigram adaugat; ipoteza: rolurile sunt locale"
    rau:   "update", "fix", "wip"
  Un commit = o idee. Istoricul git este istoria rationamentului.

## 6. Cum predai contextul unei sesiuni noi de Claude
  Da-i link-ul la repo (sau adauga repo-ul prin conectorul GitHub) si spune:
    "Citeste in ordinea asta: 00_CITESTE_INTAI.md, JURNAL.md (tot), PRINCIPII_EVALUARE.md, apoi REZULTATE.md si logs/ din ultima saptamana.
     Nu propune experimente inainte de a-mi spune ce ai inteles din JURNAL.md si ce rezultate ti se par neconfirmate."
  Daca vrei sa comunicam asincron: deschide un Issue pe GitHub per intrebare/experiment; raspunsurile si deciziile raman acolo, nu in chat.
  Etichete utile pentru Issues: `de-validat`, `evaluare`, `idee`, `blocat`.

## 7. Ramuri
  main = tot ce e verificat (evaluare conform principiilor, >=3 seed-uri). Experimentele speculative pe ramuri `exp/<nume>`;
  se unesc in main doar cu intrare in JURNAL.md si tabel in REZULTATE.md.
