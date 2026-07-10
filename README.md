# goml_fp8_train

Чистое дерево проекта под **FP8-тренировку**: sealed-ядра flash-attention backward + L-форвард и
Go-фреймворк. Без R&D-хлама (сотни экспериментальных `flash_attention_v*`, `probe_*`, `bench_*`,
старых `fa_bwd_*`, ncu/.sass/.ptx/.log) — тот живёт в исходном `goml/libs`.

## Якорь
- Источник: **goml @ commit `4732a380a817e63d8592532f87e1daf26ec9d2f8`**, тег **`w0-seal-v1`**.
- merged_v1 = `2bf32ab7` (из `runs/archive/040_sealed`), dk_new = `a9f0ded8` (033_sealed),
  dq_new = `d7a11a3d`, common = `4407ec9c`, dk = `068d6a4f`, forward train = `2cf06fd0`.
  Полный список — `kernels/MANIFEST.md5`.

## Подтверждённая скорость sealed-цепочки
- **Медиана 43.9 ms / ~401 T @16N²d** на чистом GPU (RTX PRO 6000 Blackwell), 4 прогона 43.83–43.98 ms,
  CV ≈ 0.14% — verify GREEN (совпало с ledger 041: 44.206 ms / 398 T в пределах дрейфа).
- Замер проводить только на **пустом GPU** (`nvidia-smi --query-compute-apps` пуст): конкуренция за карту
  даёт ложный ~2× даже без троттла.

## Точность стека (floor vs FP64-golden)
- non-causal **4.7e-3**, causal **3.2e-2** — это норма стека, не баг.

## Структура
```
kernels/
  backward/   fa_bwd_merged_v1.cu, fa_bwd_dk_new.cu, fa_bwd_dq_new.cu, fa_bwd_common.cuh, fa_bwd_dk.cu
              (D-precompute = kernel_d_precompute внутри fa_bwd_dk.cu)
  forward/    fa_sm120/  — пакет L-форварда; sealed train-ядро src/_v121r_train_kernel.cu (эмитит LSE)
  bench/      bench_r2c_e2e.cu + Makefile — REFERENCE-харнесс E2E-цепочки (в этой фазе не собирается)
  MANIFEST.md5, BUILD_FLAGS.md
<go-фреймворк>: tensor/ nn/ ops/ optim/ tokenizer/ train/ core/ backend/ cmd/ autograd/ + go.mod/go.sum
```

## Что НЕ включено
- Kernel-R&D (`goml/libs`, сотни экспериментов), `runs/`, `.git` исходного репо, `DISPATCHER.md` (файл opt-ветки).
- Нераспознанные top-level каталоги goml (release_*, cuda-future, docs, scripts, snapshot, ssh, libs1/2, runs1/2, ncu_home) — НЕ копировались (решение по ним за Vugar).

## ⚠️ Известный рассинхрон reference-бенча
`kernels/bench/bench_r2c_e2e.cu` — это **текущая opt-версия** харнесса (не в теге; bench untracked).
Её fingerprint-таблица ждёт `dk_new` = **124 рег** (стадия 061 S2v4), тогда как в `kernels/backward`
лежит **sealed dk_new = 128 рег** (033, a9f0ded8). Поэтому сборка этого бенча против клонированных
sealed-исходников упрётся в fingerprint-mismatch по dk_new без правки ожидаемых регистров назад на 128.
Бенч включён как провенанс-референс структуры E2E-замера; замеренные 43.9 ms/401 T получены на sealed
dk_new (128). Реальная сборка/бенч цепочки — на W1 с C ABI.
