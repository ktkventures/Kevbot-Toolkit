# V5 split verification marker

Committed to `dev` on 2026-08-01 to prove the environment split works.

**The test:** this commit must deploy to the DEV environment and must NOT reach production.
Production now tracks the `production` branch; dev tracks `dev`. Before the split,
this commit would have gone straight to the live trading fleet.

Board #165 Phase A, final verification.
