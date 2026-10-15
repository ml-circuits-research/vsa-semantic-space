# Principii de evaluare (de verificat explicit la fiecare experiment nou)

Greselile deja facute in acest proiect, si regula care le previne:

1. **Adresele de decodare vin din LUME, niciodata din model.** (Greseala rundei 1: adrese din forma activa -> penaliza adrese
   suprapuse legitime.) Lumea defineste ce e un rol; modelul doar trebuie sa ajunga acolo.
2. **Lumea si modelul impart ACELEASI identitati (F).** (Bug runda 1.) Verific: `model.F is world.F`.
3. **Sansa se calculeaza si se tipareste langa fiecare cifra.** Cu N candidati, sansa = 1/N; o cifra fara sansa langa ea nu se raporteaza.
4. **Separ ce a fost vazut de ce nu a fost vazut, pe TREI axe independente**: identitati (cuvinte noi), formulari (template-uri noi),
   TRANZITII (perechi de categorii adiacente nevazute). O "formulare noua" cu tranzitii vazute si una cu tranzitii nevazute sunt
   doua teste diferite si se raporteaza separat.
5. **Raportez marja, nu doar acuratetea**: marja = cos(candidat corect) - max cos(candidat gresit). 1.00 cu marja 0.15 si 1.00 cu marja
   0.50 sunt stari diferite.
6. **Minim 3 seed-uri, cu deviatie standard**; seed-ul schimba si lumea (identitati, vectori de rol), nu doar initializarea.
7. **Nicio calibrare din setul de test.** Tot ce e "referinta" (adrese, praguri) vine din antrenare sau din lume.
8. **Fiecare variatie de model e evaluata cu EXACT aceeasi functie de evaluare si aceleasi date de test** (aceleasi seed-uri de lume).
9. **Controlul trivial e obligatoriu**: pentru orice rezultat, intreb "ce ar da un model care ignora ordinea / ignora identitatile /
   raspunde majoritar?" si il tiparesc in acelasi tabel.
10. **Nu interpretez un esec ca imposibilitate inainte de a verifica evaluarea.** (Greseala rundei 1: "imposibilitate algebrica" care
    era un artefact de evaluare.) Ordinea verificarilor: evaluare -> date -> optimizare -> abia apoi arhitectura.
11. **Trasare pe un exemplu concret inainte de a antrena**: tiparesc adresele (lista de parametri) pentru fiecare cuvant din fiecare
    formulare si verific manual ca ce cred ca face modelul e ce face.
12. **Cifrele de antrenare (loss, cos de predictie) nu sunt rezultate**; se raporteaza doar ca diagnostic.

Checklist la finalul fiecarui script de evaluare (tiparit in log):
  [ ] adrese din lume   [ ] F comun   [ ] sansa tiparita   [ ] axe vazut/nevazut separate   [ ] marja   [ ] >=3 seed-uri   [ ] control trivial
