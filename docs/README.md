# NLDS Documentation

Short index of the docs in this repo.

## User guides

- Overview: `docs/user-guide/overview.md`
- Phase portrait + Lyapunov (Tab 1): `docs/user-guide/phase-portrait.md`
- Time series (Tab 2): `docs/user-guide/time-series.md`
- Parameter sweep (bifurcation + Lyapunov, Tab 3): `docs/user-guide/bifurcation-tab3.md`

## Greek quick guides

- Γρήγορη εκκίνηση: `docs/greek/quick-start.md`
- Tab 3 οδηγός: `docs/greek/tab3-greek.md`

## Theory notes

- Bifurcation vs continuation: `docs/theory/bifurcation-vs-continuation.md`
- Poincare sections: `docs/theory/poincare-section.md`
- Numerical techniques: `docs/theory/numerical-notes.md`

## Developer notes

- Developer README (execution map): `docs/developer/README.md`
- Architecture: `docs/developer/architecture.md`
- Sweep engine: `docs/developer/sweep-engine.md`
- Session state: `docs/developer/session-state.md`
- Performance: `docs/developer/performance.md`

## Deployment notes (Streamlit Cloud)

- Streamlit Cloud reads `requirements.txt` and `packages.txt` from the repo root,
  even when the app entry point is `app/nlds_app.py`.
