# goml_fp8_train

Чистое дерево проекта под **FP8-тренировку**: sealed-ядра flash-attention backward + L-форвард и
Go-фреймворк. Без R&D-хлама (сотни экспериментальных `flash_attention_v*`, `probe_*`, `bench_*`,
старых `fa_bwd_*`, ncu/.sass/.ptx/.log) — тот живёт в исходном `goml/libs` (kernel-creation).

## Якорь (sealed v2)
- Источник: **goml @ commit `2314cbead0fc230792bab785fca886dc584404e4`**, тег **`bwd-sealed-v2`**
  — *"seal: FP8 bwd v2 (42.35ms E2E / 415.44T proj) — 061 S2v4 dk_new"*.
- merged_v1 = `2bf32ab7`, **dk_new = `25e5e107`** (S2v4, 124 рег), dq_new = `d7a11a3d`,
  common = `4407ec9c`, dk = `068d6a4f`, forward train = `2cf06fd0`. Полный список — `kernels/MANIFEST.md5`.
- Предыдущая эпоха **`w0-seal-v1`** (`4732a380`, dk_new 128 рег) — остаётся на remote для
  reproducibility старых экспериментов, но **production sealed = v2**.

## Скорость sealed-цепочки
- **v2: 42.35 ms E2E (non-causal) / 415.44 T (proj)**, causal 22.206 ms — cert 30-run 062/063.
  Это **−4.4% wall vs v1** (v1 было 44.206 ms / 398 T; локально verify давал 43.9 ms / ~401 T).
- Замер проводить **только на пустом GPU** (`nvidia-smi --query-compute-apps` пуст): конкуренция за карту
  даёт ложный ~2× даже без троттла.

## Точность стека (floor vs FP64-golden)
- non-causal **4.7e-3**, causal **3.2e-2** — норма стека, не баг.

## Структура
```
kernels/
  backward/   fa_bwd_merged_v1.cu, fa_bwd_dk_new.cu (S2v4/124r), fa_bwd_dq_new.cu,
              fa_bwd_common.cuh, fa_bwd_dk.cu  (D-precompute = kernel_d_precompute внутри fa_bwd_dk.cu)
  forward/    fa_sm120/  — пакет L-форварда; sealed train-ядро src/_v121r_train_kernel.cu (эмитит LSE)
  bench/      bench_r2c_e2e.cu + Makefile — REFERENCE-харнесс E2E-цепочки
              (fingerprint 252/124/69/38 — с v2 сходится естественно; патч 070: dS_T dead-alloc removed)
  MANIFEST.md5, BUILD_FLAGS.md
docs/         FP8_FA_ROADMAP.md — дизайн-роадмап
<go-фреймворк>: tensor/ nn/ ops/ optim/ tokenizer/ train/ core/ backend/ cmd/ autograd/ + go.mod/go.sum
```

## Что НЕ включено
- Kernel-R&D (`goml/libs`, сотни экспериментов), `runs/`, `libs1/2`, `cuda-future`, `scripts`, `ssh`,
  релиз-снапшоты, `ncu_home` — всё это **kernel-creation сторона**, живёт в `goml`.
- `DISPATCHER.md` (файл opt-ветки).

## Разделение ответственности
- **Этот репо** = потребитель sealed-ядер (FP8-тренировка). Ядра приходят сюда **только через тег seal**.
- **`goml/libs`** = kernel-creation (churn, стадии 061/070…). Шов между сторонами — **C ABI (W1)**;
  Go-бэкенд грузит `.so` через purego/dlopen (поэтому `go build ./...` проходит без CUDA).
