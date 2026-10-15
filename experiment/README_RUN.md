# Cum rulezi

Cerinte: Python 3.10+, numpy. Atat. NU e nevoie de GPU: toate experimentele de aici ruleaza in 0.5-5 minute pe un CPU obisnuit.
(GPU devine util abia la D >= 4096, vocabular de mii de cuvinte si text real; atunci portarea in torch cu tensori complecsi e directa:
 bind = inmultire element cu element, bundle = adunare, iar gradientele le da autograd, nu mai trebuie scrise de mana.)

Fisiere, in ordinea istorica (fiecare e autonom sau importa pe cel dinainte):
- sim_deplasari.py      primele experimente (A predictie, B contrastiv, C consecinta, D prototipuri, E generativ). `python3 sim_deplasari.py`
- f.py, f2.py           factorizarea cuvantului (deplasare pe categorie); raport pe roluri. `python3 f2.py`
- camp.py               MODELUL FINAL (World, Model): pas uniform, adresare amanata, invatare generativa, evaluare pe roluri. `python3 camp.py`
- gh.py                 G (adresare amanata) + H (identitati nevazute)
- i2.py                 I: categoriile descoperite nesupervizat (clustering iterativ)
- j.py                  J: capacitate (D, numar de identitati, adjuncte)
- k.py                  K: compozitie ierarhica "X spune ca <propozitie>"
- mlp_control.py        CONTROL: MLP pe bundle fara ordine (baseline 'fara ordine, cu capacitate')
- control_transformer.py  CONTROL: transformer mic (numpy, backprop manual, gradient-check inclus) vs VSA pe aceeasi sarcina
- v2.py, v3.py, gates.py, nest2.py, v4.py   RUNDA 2-3: asteptari, dual, porti invatate, imbricare fara reguli, adrese redundante
- PRINCIPII_EVALUARE.md  reguli de evaluare (de citit inainte de orice experiment nou)
- REZULTATE.md          toate rezultatele si concluziile
- INTERPRETARE.md       cum se citeste fiecare metrica, ce ar infirma teza, capcane in cifre
- POZITIONARE.md        nume, figura 1, trei articole posibile, bariera, controale, ce sa citezi
- README_DGX.md         planul de validare pe DGX: experimente cu criterii merge/nu merge, surse de date, portare torch
- log_*.txt             iesirile brute ale rularilor

Parametrii principali: D (dimensiunea hipervectorilor), n_sent (propozitii de antrenare), lr (Adam, 0.02 e bun), persons/objects.

Urmatorii pasi sugerati (in ordinea valorii):
1. Invatarea declansatoarelor: "verbul scrie bufferul" si "ca deschide cadru" sunt puse de mana in Model.run / NestedModel.run.
   Ipoteza: ambele sunt puncte unde eroarea de predictie a nivelului rapid sare; de testat pe lumea sintetica.
2. Text real intr-un domeniu inchis; consecinta = starea propozitiei urmatoare (self-supervised) in loc de lumea sintetica.
3. Capacitate: D=4096-16384 si coduri rare pe blocuri (sparse block codes) pentru cleanup mai curat la vocabular mare.
4. Control mai puternic: transformer mai mare + mai multe date, ca sa vezi unde se inchide diferenta de eficienta.
