# Γρήγορη εκκίνηση (NLDS)

Σύντομος οδηγός για να τρέξεις την εφαρμογή.

---

## Βήματα

1) Εκτέλεσε: `streamlit run app/nlds_app.py`.  
2) Διάλεξε σύστημα (Lorenz/Rossler/Custom) και ρυθμίσεις ολοκλήρωσης από το sidebar.  
3) Ρύθμισε παραμέτρους συστήματος (sidebar) και άξονες στο Tab 1.  
4) Εξερεύνησε τα tabs: Phase portrait (μαζί με Lyapunov και Poincaré map), Time series, Parameter Sweep Analysis (διακλάδωση + Lyapunov).  
5) Κατέβασε CSV από το tab Export (τροχιές, διακλάδωση, Lyapunov).
6) Άνοιξε Help (Eng) / Help(Ελλ) για το πλήρες manual.

## Συμβουλές

- Μείωσε `dt` για ομαλότερες καμπύλες, αύξησε `final time` για μεγαλύτερη διάρκεια.  
- Χρησιμοποίησε transient cut για να αγνοήσεις αρχικές μεταβατικές καταστάσεις.  
- Το Poincaré map στο Tab 1 βγαίνει από την ήδη υπολογισμένη τροχιά και έχει δικό του `Max points` για display decimation.  
- Αν δεις σφάλμα, έλεγξε το μήκος `y0` στα `initial` και σωστή γραφή εξισώσεων/παραμέτρων.  
- Τα parallel workers δουλεύουν μόνο τοπικά και μόνο σε independent mode (όχι Continuation).
- Για RK4 ή Symplectic Forest-Ruth, το Numba ενεργοποιείται αυτόματα όταν είναι διαθέσιμο.
