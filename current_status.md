# Stare curenta (actualizat la fiecare punct de judecata; ultima: 2026-10-15 seara, commit 965332d)

## Ce s-a facut azi pe DGX
- Mediu: .venv cu numpy 2.5.4 si torch 2.14.1+cu130 (aarch64 + CUDA, GPU GB10 vizibil).
- Curatenie documente (a1ff20b): aliniere la runda 2-3; retrase "imposibilitatea algebrica" si "adresarea amanata"; nivel de incredere corectat.
- Runda 4: v4.py numpy pe CPU DGX reproduce EXACT runda 3 (3 decimale). v4_torch.py (GPU) da scoruri IDENTICE cu numpy pe toate 24 rulari.
  Timp/varianta: CPU vechi ~77 s -> CPU DGX 24.8 s -> GPU batch=1 28.5 s (nu castiga: Python) -> GPU encodare o data + batch 32: 2.2 s.
  Batch 8-32 pastreaza concluziile pentru "toate 4" (+-0.03); codurile slabe pierd la batch 32.
- Runda 5 (§5.1): v5_scale.py, lume de 3x (6 cadre x 16 formulari, 18+18 identitati, imbricare 0-3), 5 seed-uri, D=1024, 30k, batch 8.

## Rezultate runda 5 (5 seed-uri)
| cod | vazute | id. noi | form. noi tranz. vazute | tranz. NEvazute | imbricare ad.1-2 | ad.3 |
|---|---|---|---|---|---|---|
| path+chain+bigram | 1.000 | 1.000 | 0.998 | 0.821 | 0.938 | 0.673 |
| toate 4 | 1.000 | 1.000 | 0.841 | 0.984 | 0.943 | 0.687 |
| (lume mica, runda 3) path+chain+bigram | 1.000 | 1.000 | 0.987 | 0.824 | — | — |
| (lume mica, runda 3) toate 4 | 1.000 | 1.000 | 0.942 | 0.901 | — | — |

## Concluzii fata de scop
1. REPRODUS: path+chain+bigram e stabil intre lumi (0.99 / 0.82). Identitati noi 1.00. Acesta e modelul care tine.
2. NU reprodus: "toate 4" se inverseaza. Cauza: codul pos = prior de pozitie absoluta. Invata din T1 "persoana de la pozitia 5 = agent";
   in T10 destinatarul e la 5 => pos voteaza complet gresit (A->C, 0.15) si bate adresele suprapuse ale primului cuvant. Castigul lui pe
   tranzitii nevazute vine din categorii noi ('catre', 'il') unde celelalte coduri n-au nimic. Ambele tin de setul de formulari, nu de mecanism.
   PROPUNERE: raportam modelul = path+chain+bigram; pos separat, ca prior.
3. NEGATIV SI CURAT (cel mai important): imbricarea de adancime >= 2 e nerezolvabila cu adrese din prefix.
   '{X1} spune ca {A} V ...' si '{X1} spune ca {X2} zice ca ...' au prefix identic 'P spune ca P' => orice cod cauzal da acelasi loc pentru
   A (ad.1) si X2 (ad.2), lumea cere roluri diferite => ~0.5 (X2 0.28-0.53 pentru orice cod, confundat cu A), CHIAR pe date vazute.
   Nu ajuta D, nu ajuta datele. Runda 2 a retras "adresarea amanata" prea mult: nu e nevoie de "verbul scrie bufferul", dar E nevoie ca un
   cuvant sa poata fi reasezat de ce urmeaza. Prima dovada fortata de date. Schimba figura 1 si contributiile articolului.
4. Defect de design prins: la adancime 3 apare rolul X3, niciodata tinta la antrenare => la sansa prin constructie (0.08 ~ 1/12);
   cifra "ad.3" il amesteca cu degradarea reala a lui A/X1/X2. Devenit regula 14 in PRINCIPII_EVALUARE.
   Reguli noi: 13 (acelasi regim de optimizare la comparatii), 15 (detaliere per formulare x rol inainte de interpretare).

## Intrebari pentru judecata celeilalte sesiuni
1. Acceptam path+chain+bigram ca "modelul" si pos ca prior separat?
2. Pasul urmator: §5.1b (reasezare retroactiva: un cuvant plasat poate fi mutat cand apare un emitator nou; criteriu X2 > 0.9 la adancime 2
   fara pierderi altundeva) INAINTE de §5.2 (porti prin gradient)? Argument: portile depind de acelasi mecanism de asteptari.

## Fisiere relevante
experiment/REZULTATE.md (runda 4, 5) | experiment/JURNAL.md | experiment/PRINCIPII_EVALUARE.md (13-15) | experiment/00_CITESTE_INTAI.md §4-5
experiment/v4_torch.py, v5_scale.py, diag_v5.py | experiment/logs/2026-10-15_1128_v5_scale_0defffd.txt
