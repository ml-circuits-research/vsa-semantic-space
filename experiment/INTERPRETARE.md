# Cum se citesc rezultatele (si ce le-ar infirma)

## Metricile, una cate una

**Decodarea rolurilor (camp.py, k.py, control_transformer.py).**
Se ia starea S a unei formulari oarecare (pasiv, "primeste de la", cu dativ in fata...). Se de-binda cu ADRESA rolului
luata din FORMA ACTIVA (template 0) si se face cleanup printre candidati (persoane sau obiecte). Corect = iese identitatea reala.
- De ce asa: masoara direct teza "drumuri diferite, acelasi punct": daca agentul din pasiv e citibil la adresa agentului din activ,
  cele doua drumuri au ajuns in acelasi loc PENTRU ACEL ROL. E mai strict decat cosinusul global intre stari (care poate fi mare
  din cauza cuvintelor comune) si mai informativ (spune CARE rol a esuat).
- Sansa = 1/numar de candidati (0.125 la 8, 0.25 la 4, 0.083 la 12). Orice sub ~2x sansa inseamna "nimic".
- Template 0 da mereu 1.00 prin constructie (adresele vin de acolo); NU e un rezultat, e calibrarea. Cifrele care conteaza sunt T1-T4.
- "Media pe toate formularile si rolurile" include T0, deci e usor umflata; scade ~0.1 daca excluzi T0.

**Cos(pred, S_lume) si calitatea predictiei.** Cat de bine genereaza nivelul lent starea consecintei. E obiectivul antrenat; nu e o
masura a echivalentei. Experimentul C a aratat ca poate fi perfect (100% recuperare) cu echivalenta zero.

**Similaritate parafraze / roluri schimbate / aleator (sim_deplasari.py).** Cosinus intre stari finale. Cifra singura nu spune nimic:
conteaza DIFERENTA parafraze minus roluri-schimbate (negativ greu) si "parafraza bate schimbul" (fractia perechilor unde parafraza e
mai aproape decat versiunea cu rolurile inversate). Aleator ridicat (>0.3) = colaps: modelul a facut totul similar.

**Deplasare medie pe clasa** (1 - cos mediu al fazelor): 1.0 = deplasare maxima (aleatorie), 0 = identitate. Testa H1 (functionalele
mici, verbele mari). Nu s-a confirmat decat slab; nu insista pe ea.

**Spearman cu scorul uman (sick_test.py).** Corelatia de rang intre cosinusul starilor si inrudirea 1-5. Baseline-ul decisiv e Jaccard
(suprapunere de cuvinte): 0.575. Orice model sub el a pierdut informatie lexicala fara sa castige structura. SICK NU masoara roluri
(29 din 4927 perechi au aceleasi cuvinte in alta ordine); un rezultat bun pe SICK nu ar fi confirmat teza, unul slab nu o infirma.

**WikiSQL (wikisql_test.py).** Decodare din predictie: coloana selectata (candidati = coloanele tabelului, sansa ~0.15), agregator
(6 clase, majoritarul 0.72: orice sub 0.72 pe agregator e MAI RAU decat a raspunde mereu "none"), coloana conditiei. Modelul nu vede
tabelul, deci coloana e ghicita doar din cuvintele intrebarii; asta plafoneaza rezultatul. "Acelasi schelet / schelet diferit" = cosinus
intre predictii pentru intrebari din acelasi tabel cu acelasi (sel, agg, coloane-conditii) vs diferit; diferenta conteaza, nu valorile.

## Cele trei rezultate care sustin teza si ce le-ar darama
1. Factorizarea e necesara (G3 vs G2: 0.50 vs 0.86; E2 demonstrat algebric). Daramat de: un model cu deplasare per cuvant care ajunge
   la >0.8 cu alta regula de invatare.
2. Predictia generativa (operatie fixa) produce echivalenta, cea asociativa nu (C,D vs F,G). Daramat de: o memorie asociativa cu gat de
   sticla (capacitate limitata) care aliniaza starile la fel de bine. (Experiment util de facut, chiar.)
3. Categoriile emerg (I: clustere exacte; SICK: man/person/woman). Daramat de: clustere la fel de curate obtinute din simpla co-aparitie
   (word2vec pe acelasi corpus). Fa acest control; daca word2vec le da la fel, emergenta nu e meritul arhitecturii.

## Capcane cunoscute in cifrele de aici
- Seed unic peste tot. Variatia intre seed-uri NU a fost masurata; pe sintetic te astepti la +-0.05, pe WikiSQL la +-0.02. Ruleaza 5 seed-uri.
- Lumea si modelul trebuie sa imparta aceleasi trasaturi F (bug gasit tarziu; experimentele G-K au ruat cu F diferite, controlul cu F comune).
  Cifrele G-K sunt deci subestimari; re-ruleaza-le cu F comune inainte de a le cita.
- Evaluarea pe sintetic foloseste aceeasi lume la antrenare si test (doar identitatile/formularile sunt tinute deoparte). Nu exista "test set"
  in sensul clasic pentru cadre: ambele cadre au fost vazute.
- Controlul cu transformer: fara tuning de hiperparametri, 3 epoci-echivalent, fara dropout/weight decay. Un transformer bine reglat la 40k
  exemple ar generaliza mai bine la formulari noi (nu si la identitati noi, unde embeddingul invatat e structural dezavantajat; dar cu
  trasaturi fixe la intrare, cum i-am dat, nici acolo nu e exclus cu mai multe date).

## Adaugat dupa runda 2
- Decodarea cu adresele lumii (nu ale formei active) e singura corecta: adresele formei active penalizeaza adresele suprapuse, care sunt
  LEGITIME (argmax le rezolva). Toate cifrele din runda 1 pe formulari neactive sunt subestimari din acest motiv.
- Raportati MARJA pe langa acuratete: 1.00 cu marja 0.15 (drum) si 1.00 cu marja 0.50 (asteptari) sunt stari foarte diferite; la
  vocabular mare marja mica devine eroare.
- "Formulare noua" testeaza tranzitii neantrenate; in adresarea multiplicativa o singura faza neantrenata randomizeaza adresa. Separati
  in raport: formulari noi cu tranzitii vazute vs cu tranzitii nevazute.
