# Changelog

## [0.3.1](https://github.com/moodsona/moodsona/compare/v0.3.0...v0.3.1) (2026-11-20)


### Bug Fixes

* make moodsona --version print the version instead of a usage error ([#55](https://github.com/moodsona/moodsona/issues/55)) ([8228a1a](https://github.com/moodsona/moodsona/commit/8228a1a1dfd4c2f3b04d2422babc3fd6c958eedd))

## [0.3.0](https://github.com/moodsona/moodsona/compare/v0.2.0...v0.3.0) (2026-11-20)


### ⚠ BREAKING CHANGES

* run() now takes the persona and the companion by position and everything else by keyword: run(persona, companion, *, user_model, days=3, turns_per_day=6, rng_seed=0, ...). user_model has no default.

### Features

* add rule probes to check that each persona rule can fire ([#51](https://github.com/moodsona/moodsona/issues/51)) ([3e32cfa](https://github.com/moodsona/moodsona/commit/3e32cfa42baf8d06be7f5407036dbba0852124d2))
* add SimulatedUser, a persona with the same reply(history) method as a companion ([#41](https://github.com/moodsona/moodsona/issues/41)) ([b71f136](https://github.com/moodsona/moodsona/commit/b71f136a2030a95360a8de86970f78ef2f396f22))
* give repeated reaction rules diminishing returns within a day ([#47](https://github.com/moodsona/moodsona/issues/47)) ([28b2883](https://github.com/moodsona/moodsona/commit/28b2883fcf3f6217dcedec9d83d3e47e69fdcaed))
* make run() take the persona and companion first and everything else by keyword ([#44](https://github.com/moodsona/moodsona/issues/44)) ([c02f025](https://github.com/moodsona/moodsona/commit/c02f025496d41091a886a153d69589a5c1ae5556))
* record a format version in transcripts ([#52](https://github.com/moodsona/moodsona/issues/52)) ([6b561ce](https://github.com/moodsona/moodsona/commit/6b561ce47603831147e180b512a0535ae0e65986))
* summarise runs by mean emotion and add repeat helpers ([#50](https://github.com/moodsona/moodsona/issues/50)) ([6d71584](https://github.com/moodsona/moodsona/commit/6d7158437efca374398daafab307df143cf607a8))
* tag every message with its simulated day so companions can tell days apart ([#45](https://github.com/moodsona/moodsona/issues/45)) ([9db2274](https://github.com/moodsona/moodsona/commit/9db22743b7477b589c3b130266f635c290da2c18))


### Bug Fixes

* **persona:** add rules for unrequested advice lists, stacked questions and early referrals ([#49](https://github.com/moodsona/moodsona/issues/49)) ([1292bc1](https://github.com/moodsona/moodsona/commit/1292bc15f6af03a73317dfac20dc24997d203c6a))
* **persona:** keep boundary-tester on friendship and exclusivity, away from disappearance ([#46](https://github.com/moodsona/moodsona/issues/46)) ([bb6eb67](https://github.com/moodsona/moodsona/commit/bb6eb675e3f56cd466a9f5bc2d3b6a028b9c6c43))

## [0.2.0](https://github.com/moodsona/moodsona/compare/v0.1.0...v0.2.0) (2026-10-25)


### Features

* accept bundled persona ids in `moodsona smoke` and `validate` ([#8](https://github.com/moodsona/moodsona/issues/8)) ([9f065e3](https://github.com/moodsona/moodsona/commit/9f065e3d5505b36258bcdc02dc321f9836e97e97))
* add `moodsona bench`, a scaffold for the user-model leaderboard ([#11](https://github.com/moodsona/moodsona/issues/11)) ([3a36d92](https://github.com/moodsona/moodsona/commit/3a36d9241092d828878d04f3298aa9c0fcbf8780))
* add an experimental classifier two-sample fidelity check ([#12](https://github.com/moodsona/moodsona/issues/12)) ([ef12e3d](https://github.com/moodsona/moodsona/commit/ef12e3d085e9146f059f337225b3ffe994c49370))
* add transcript metrics and `moodsona report` ([#9](https://github.com/moodsona/moodsona/issues/9)) ([0db8227](https://github.com/moodsona/moodsona/commit/0db8227b6453c79a1e7124333ece098403b8f736))

## 0.1.0 (2026-10-24)


### Features

* initial moodsona v0.1 ([880849c](https://github.com/moodsona/moodsona/commit/880849c85088009e37b0ed60a2d58495aa4614f9))
