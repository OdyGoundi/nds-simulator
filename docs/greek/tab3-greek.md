# Σάρωση Παραμέτρου (Tab 3) – Διακλάδωση & Lyapunov

Μίνι οδηγός για σάρωση Poincare, διακλάδωση και Lyapunov.

---

## Διακλάδωση (αριστερή στήλη)

1) Επίλεξε παράμετρο σάρωσης, αρχή/τέλος/βήμα.  
2) Διάλεξε τομή Poincare (μεταβλητή/τιμή) και κατεύθυνση crossing.  
3) Διάλεξε sweep mode: **Bifurcation (reset ICs)** ή **Continuation (warm start)**.  
4) Προαιρετικά: **Parallel sweep (local only)** και **Workers**.  
5) Πάτησε **Generate Bifurcation Diagram** ή **Continue Bifurcation**.

## Lyapunov (δεξιά στήλη)

1) Ίδια παράμετρος σάρωσης από το πάνω μέρος.  
2) Ρύθμισε **QR interval (time)** και **Transient fraction**.  
3) Προαιρετικά: **Parallel sweep (local only)** και **Workers**.  
4) Πάτησε **Generate Lyapunov Diagram** ή **Continue Lyapunov**.  
5) Το διάγραμμα δείχνει lambda vs παράμετρο (μωβ γραμμές = όρια συνέχισης).

## Συμβουλές

- Continuation = ομαλή συνέχεια αλλά σειριακό (χωρίς parallel).  
- Μικρό `tf_sweep` + λίγα `max_hits` για γρήγορο preview.  
- Κάνε reset όταν αλλάζεις βασικές ρυθμίσεις (παράμετρος, τομή, μέθοδος).

## Σημειώσεις για λύτη

- Οι σαρώσεις χρησιμοποιούν τον solver που έχεις επιλέξει στο sidebar.  
- Τα event-based crossings δουλεύουν μόνο για IVP solvers (RK45/DOP853) + crossing.  
- RK4 και Symplectic Forest‑Ruth είναι fixed‑step· το Numba ενεργοποιείται αυτόματα όταν είναι διαθέσιμο.  
- Οι expression-based τομές απενεργοποιούν το fast Numba sweep path για built‑in συστήματα.
