
Приложение получает аудиофайл (wav, flac) делает wavelet преобразование на лету с помощью gpu cuda и отображает данные графически на дисплее. 
Для стерео - выводится два изображения.
Необходимо делать преобразование эффективно на gpu. Приложение должно позволять масштабировать полученные данные с помощью мыши в широком диапазоне(позволять рассмотреть очень мелкие детали преобразованных звуковых данных). Необходимо иметь возможность настраивать параметры wavelet-преобразования. Поскольку данных после преобразования может быть очень много, то следует сделать оптимизацию - чтобы корректно отображались данные только под определённый масштаб отображения. При масштабировании(увеличении мелких деталей) данные просчитывались более подробно.




## Requirements

| Component        | Requirement                                    |
|------------------|------------------------------------------------|
| OS               | Fedora Linux (Fedora 44)                       |
| GPU              | NVIDIA RTX 2060 Super (sm_75, ≥8 GiB VRAM)     |
| CUDA Toolkit     | 13.1 at `/usr/local/cuda-13.1`                 |
| Host C++ compiler| g++ / gcc (16, 15, 14, or 13 — auto-detected)  |
| Rust             | stable 2021 edition                            |


### Флаги nvcc
```
--gpu-architecture=sm_75
--allow-unsupported-compiler    # gcc-16 новее, чем официально поддерживает nvcc 13.1
-std=c++14                      # cicc (внутренний LLVM) падает на заголовках libstdc++16 c++17
--use_fast_math
-O2
```
**export CUDA_ROOT=/usr/local/cuda-13.1**

`nvcc` is invoked **at runtime** (not build time) to compile
`src/kernel.cu → src/kernel.ptx`.  

No `build.rs` required — compilation happens in `main()`.
