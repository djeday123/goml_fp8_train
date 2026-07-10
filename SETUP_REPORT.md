# SETUP_REPORT — чистый клон goml_fp8_train под FP8-тренировку

**Итог:** ✅ **ВЫПОЛНЕНО.** Дерево собрано, sealed-ядра сверены (все md5 PASS), Go-фреймворк собирается
(`go build ./...` EXIT=0), клон закоммичен локально. Провенанс: **goml @ `4732a380`, тег `w0-seal-v1`**.
Commit клона: **`fb00d1dd2a74f74f35831d47ba16c658ee38c086`**.

---

## ARTIFACT HEADER
```
# ls -la /data/lib/podman-data/projects/goml_fp8_train/
drwxr-xr-x  autograd backend cmd core nn ops optim tensor tokenizer train   (go-фреймворк)
drwxr-xr-x  kernels/          (backward + forward + bench + MANIFEST + BUILD_FLAGS)
-rw-r--r--  go.mod (89)  go.sum (175)  README.md (3378)  .gitignore (36)
drwxr-xr-x  .git/             (клон-репо, commit fb00d1d)

# ls -la kernels/backward/
-rw-r--r-- 13342  fa_bwd_common.cuh
-rw-r--r-- 25409  fa_bwd_dk.cu
-rw-r--r-- 19919  fa_bwd_dk_new.cu
-rw-r--r-- 18834  fa_bwd_dq_new.cu
-rw-r--r-- 25638  fa_bwd_merged_v1.cu
# forward пакет fa_sm120: 640K ;  файлов в git: 87
```

## Gate 0 — среда + тег
- HEAD=`4732a380`, `w0-seal-v1^{}`=`4732a380`, merged-из-тега=`2bf32ab7` ✅.
- Целевой путь `/data/lib/podman-data/projects/goml_fp8_train` — запись через Bash OK (после добавления в allowed-dirs).
- CUDA 13.1.1 (`/usr/local/cuda-13.1`), driver 580.159.03, GPU RTX PRO 6000 Blackwell (sm_120a), Go 1.24.11.

## Фаза 2 — md5-verify backward (РОВНО ПОСЛЕ КОПИИ) — PASS
| файл | источник | эталон | клон | verdict |
|---|---|---|---|---|
| fa_bwd_merged_v1.cu | **archive/040_sealed** (opt-immune) | 2bf32ab7 | `2bf32ab7` | ✅ |
| fa_bwd_dk_new.cu | **archive/033_sealed** (не libs/!) | a9f0ded8 | `a9f0ded8` | ✅ |
| fa_bwd_dq_new.cu | libs/ (совпал с тегом) | d7a11a3d | `d7a11a3d` | ✅ |
| fa_bwd_common.cuh | libs/ | 4407ec9c | `4407ec9c` | ✅ |
| fa_bwd_dk.cu | libs/ (D-precompute внутри) | 068d6a4f | `068d6a4f` | ✅ |

**Отклонение от буквы ТЗ (обосновано):** `dk_new` взят из архива `033_sealed`, а НЕ из рабочей `libs/`,
т.к. на момент клона opt-ветка увела `libs/fa_bwd_dk_new.cu` на стадию 061 (`25e5e107`, 124 рег).
Sealed для `w0-seal-v1` = 033 (`a9f0ded8`, 128 рег). merged аналогично взят из `040_sealed` (opt-immune).

## Forward — тег==копия + L-эмиссия
- `fa_sm120/src/_v121r_train_kernel.cu`: тег `2cf06fd0` == копия `2cf06fd0` ✅ (не дрейфанул).
- L-эмиссия подтверждена: `L_out` буфер, `L = m_i + log(l_i)`, `(rmax+log2f(rsexp))*LN2` (стр. 693).
- Пакет `fa_sm120/` скопирован целиком (640K: src, include, go, python, test, Makefile, libfa_sm120.{so,a}).

## Bench — цепочка подтверждена (⚠️ рассинхрон fingerprint)
`kernels/bench/bench_r2c_e2e.cu` (+ Makefile) зовёт полную цепочку `d_precompute+merged+dk_new+dq_new`.
Бенч НЕ в теге (untracked R&D) — скопирована текущая opt-версия. Её fingerprint ждёт `dk_new`=**124 рег**
(061 S2v4), а в клоне sealed `dk_new`=**128 рег** (033). Сборка бенча против клона упрётся в fingerprint
без правки регистров назад на 128. Включён как reference-провенанс; **в этой фазе не собирался** (по ТЗ).
Замеренные 43.9 ms/401 T получены на sealed dk_new (128).

## Фаза 3 — Go-фреймворк
Скопированы allowlist-пакеты: `tensor nn ops optim tokenizer train core backend cmd autograd` + `go.mod`/`go.sum`.
**Исключено:** `libs/`, `runs/`, `.git/`, `DISPATCHER.md` (файл opt-ветки). `data/` — отсутствует в goml top-level.

**Нераспознанные top-level каталоги goml (НЕ копировал — решение за Vugar):**
`release_verify_clone`, `release_v0.2.0`, `cuda-future`, `docs`, `scripts`, `snapshot`, `ssh`, `.vscode`,
`libs1`, `libs2`, `runs1`, `runs2`, `ncu_home`.

## Фаза 5 — верификация
- md5 backward+forward из клона == эталон (таблица выше) — PASS.
- `go build ./...` → **EXIT=0**, без вывода (чисто). CUDA-бэкенд грузится через purego/dlopen — сборка
  не требует CUDA/cgo-линковки. cgo/CUDA-ошибки НЕ возникло.
- Финальное дерево (maxdepth 2): go-пакеты + kernels/{backward,forward,bench} + MANIFEST/BUILD_FLAGS/README.

## Фаза 6 — git-init клона
- `.gitignore` (*.log, *.o, *.bin, bench_r2c_e2e, /tmp/), `git init` + `git add -A` + commit.
- **Commit-hash клона: `fb00d1dd2a74f74f35831d47ba16c658ee38c086`** — якорь клона.
- `git status` чист; 87 файлов; в индексе нет DISPATCHER/libs/runs/flash_attention_v.
- **Remote НЕ создавал / НЕ пушил** — нужен новый GitHub-репо, решение Vugar.

## Границы соблюдены
Исходники не редактировал. `goml/` не менял (только чтение). Ядра не собирал/не бенчил. opt-версию merged
(`ca064452` в libs/) и opt-версию dk_new (061) НЕ использовал. Правок для сборки не потребовалось.

**Дата (UTC):** 2026-07-10.
