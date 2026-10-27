# Arc B580 Setup — Qwen3.8-27B-Flash-MTP

## Hardware & Software

| Component | Tested with |
|---|---|
| GPU | Intel Arc B580 (12 GB, Xe2 Battlemage) |
| Driver | 32.0.101.9150+ |
| CPU | AMD Ryzen 5 5600X (any x64 works) |
| OS | Windows 10/11 |
| oneAPI | 2025.3 (compiler, MKL, oneDNN, TBB) |
| MSVC | Build Tools 14.44+ |
| CMake | 3.28+ with Ninja |

## Build

```bat
@echo off
rem === VS Build Tools ===
call "C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

rem === Intel oneAPI — call each component directly (setvars.bat broken on some setups) ===
call "C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\tbb\2022.3\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\dnnl\2025.3\env\vars.bat"

cmake -B build-sycl -G Ninja ^
  -DGGML_SYCL=ON ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_SYCL_TARGET=INTEL

cmake --build build-sycl -j --target llama-server llama-bench llama-cli
```

> Use `icx` for both C and CXX — `icpx` is misdetected as GNU by CMake on Windows.

## Run

```bat
build-sycl\bin\llama-server.exe ^
  -m path\to\Qwen3.8-27B-Flash-MTP.gguf ^
  -c 131072 -ngl 99 -np 1 ^
  --spec-type draft-mtp,ngram-mod ^
  --spec-draft-n-max 4 ^
  --spec-ngram-mod-n-max 256 ^
  -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 ^
  -ub 1024 -b 2048 ^
  --temp 0.6 --top-k 20 --top-p 0.95 ^
  --jinja ^
  --host 127.0.0.1 --port 21021
```

## Performance

| Metric | Value |
|---|---|
| Generation (prose) | 54.3 t/s |
| Generation (code) | 64.7 t/s |
| Prompt processing (4k warm) | 945 t/s |
| MTP draft acceptance | 43–65% (content dependent) |
| Mean draft length | ~3.0 tokens |
| VRAM | ~9.5 GB with q4_0 KV at 131k context |

Speculative decoding: MTP draft + ngram modifier. Code gets higher acceptance due to repetition patterns.

## Cherry-Picked Patches

18 commits from [Torchit1/llama.cpp](https://github.com/Torchit1/llama.cpp) feature branches:

| Series | Commits | Effect |
|---|---|---|
| DPAS Flash Attention | 5 | Quantized KV-cache FA on XMX. **+25% gen speed** (43→54 t/s) |
| MTP batch-order fix | 1 | Correct token-indexed nextn row ordering after ubatch split |
| Reorder dequant | 1 | Coalesced q4_0/q8_0 dequantize for reorder layout |
| XMX prompt GEMM | 1 | Quantized prompt GEMMs on XMX via `GGML_SYCL_DYNAMIC_PRECISION` |
| XMX MMQ (oneDNN) | 8 | q4_0/q8_0 through oneDNN u4/s8 with vectorized activation conversion |
| P2P device alloc | 1 | PTQ1_0/PQ2_0 and FA scratch buffers via `ggml_sycl_malloc_device` |
| Upstream stubs | 1 | Glue for functions referenced by cherry-picks but absent on arc-b580 |

The XMX-MMQ/reorder-dequant patches are ready for q4_0/q8_0 quants but don't activate on PTQ1_0.

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `GGML_SYCL_DYNAMIC_PRECISION` | `F16` | XMX precision for prompt GEMMs (F16/BF16/TF32/F32) |

## Troubleshooting

- **setvars.bat fails:** Call each component's `vars.bat` individually (compiler, mkl, tbb, dnnl).
- **CMake detects icpx as GNU:** Use `icx` instead of `icpx` for both compilers.
- **Out of VRAM:** Reduce `-c` (context length) or switch KV cache to q8_0.
