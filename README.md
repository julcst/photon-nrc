![](docs/images/teaser.webm)

# PhotonNRC

This repo contains the source code for our paper [Photon-Driven Neural Radiance Caching](https://cg.cs.uni-bonn.de/publications/stamm-2026-photon).
The framework is forked from [Falcor](https://github.com/nvidiagameworks/falcor), see there for installation instructions.
Because the code uses tiny-cuda-nn and heavily utilizes raytracing and tensor hardware, a recent NVIDIA GPU is required to run it.

## Renderpasses
We provide various custom render passes for Neural Radiance Caching, Photon Mapping and the combination of both:

Pass | Description
--- | ---
[AccumulatePhotonsRTX](Source/RenderPasses/AccumulatePhotonsRTX) | Uses hardware accelerated BVH traversal and stochastic reservoirs to perform high performance inverse radius search for photon mapping. More details in the linked paper; Based on [Kern et al.](https://jcgt.org/published/0012/01/01/paper.pdf)
[DebugQueryBuffer](Source/RenderPasses/DebugQueryBuffer) | Writes query position/normal/etc from a buffer to a texture for debugging purposes.
[NRC](Source/RenderPasses/NRC) | Takes a buffer with inference and training queries and performs the online update and inference step of the [Neural Radiance Cache](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing). Based on [tiny-cuda-nn](https://github.com/nvlabs/tiny-cuda-nn)
[PathTracerQuery](Source/RenderPasses/PathTracerQuery) | A tweaked version of [PathTracer](Source/RenderPasses/PathTracer) that allows for multisampling and storing into buffers
[QuerySubsampling](Source/RenderPasses/QuerySubsampling) | Draws a random subsample of a query buffer to be used for training; Also handles query caching.
[TracePhotons](Source/RenderPasses/TracePhotons) | Traces photons from light sources and emits photon position/normal/etc into a buffer.
[TraceQueries](Source/RenderPasses/TraceQueries) | Traces queries from the camer and emits query position/normal/etc into a buffer.
[VisualizePhotons](Source/RenderPasses/VisualizePhotons) | Visualizes photons as pixels for debug purposes
[VisualizeQueries](Source/RenderPasses/VisualizeQueries) | Visualizes photon queries as disk for debug purposes

## Scripts
The benchmarks from the paper are implemented in [Source/Testbeds/PhotonNRCPaper.cpp](Source/Testbeds/PhotonNRCPaper.cpp).
We also provide a [Mogwai script](scripts/PhotonNRCPaper.py) for running the different NRC flavors.

Code was developed and tested on Linux on an RTX 5070 Ti.

## Citation
If you build on this repo in your research, please cite our paper.
The BibTex entry is

```bibtex
@inproceedings{stamm2026photon,
  author    = {Stamm, Julian C. and Kneiphof, Tom and Klein, Reinhard},
  title     = {Photon-Driven Neural Radiance Caching},
  booktitle = {Companion Proceedings of the Symposium on Interactive 3D Graphics and Games (I3D Companion '26)},
  year      = {2026},
  month     = {5},
  pages     = {3},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3807895.3807923},
  isbn      = {979-8-4007-2667-5/2026/05},
  location  = {San Francisco, CA, USA},
  keywords  = {real-time, global illumination, radiance caching, photon mapping}
}
```
