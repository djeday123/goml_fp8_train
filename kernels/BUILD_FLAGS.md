# BUILD_FLAGS — sealed-ядра goml_fp8_train

Тулчейн и флаги, которыми собиралась/сертифицировалась sealed-цепочка (провенанс w0-seal-v1 / 4732a380):

```
nvcc:  /usr/local/cuda-13.1/bin/nvcc   (CUDA 13.1.1; на PATH может быть 11.5 — использовать 13.1 явно)
arch:  -gencode arch=compute_120a,code=sm_120a
std:   -std=c++17
opt:   -O3
ptx:   -Xptxas=-v
dbg:   -lineinfo
```

## Примечания
- Отдельных Makefile для сборки `merged` / `dq_new` как самостоятельных единиц **нет** — реальная сборка
  ядер придёт с C ABI на этапе **W1**. Здесь ядра лежат как исходники (sealed), не собираются.
- Единственный собираемый артефакт-референс — `bench/bench_r2c_e2e.cu` + `bench/Makefile.bench_r2c_e2e`
  (nvcc-13.1, sm_120a, -O3, -Xptxas=-v -lineinfo). В ЭТОЙ фазе НЕ собирается.
- GPU таргет: NVIDIA RTX PRO 6000 Blackwell, sm_120a, driver 580.159.03, power 600 W.
