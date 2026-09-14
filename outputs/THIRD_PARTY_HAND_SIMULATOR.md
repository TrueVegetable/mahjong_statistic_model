# Hand-simulator dependencies

The default native CMake build fetches
[tomohxx/necessary-and-unnecessary-tiles](https://github.com/tomohxx/necessary-and-unnecessary-tiles)
at commit `e5cd6e124218808da07d76dc852bdcc4877de791`. It is GPL-3.0. The
resulting native binary links that library and is therefore distributed under
GPL-3.0 terms. Its generated `index_dw_s.bin` and `index_dw_h.bin` lookup tables
remain in the build directory and are loaded once at startup.

The local patch `patches/necessary_tiles_sequential_policy.patch` only makes the
upstream table generator compile on standard libraries that do not implement
C++ parallel execution policies. It does not change the lookup algorithm or
table contents.

With `-DHAND_EFFICIENCY_USE_TILE_MASKS=OFF`, CMake instead fetches
[Cryolite/nyanten](https://github.com/Cryolite/nyanten) at commit
`581aa72ce53e0ffeb0ef8d1ed14df2521ca5fb9b`. Its calculator headers are
MIT-licensed. Nyanten's separately GPL-licensed correctness and baseline test
files are not compiled or distributed by this project.

The Python reference environment is pinned separately in
`requirements-hand-simulator.txt`. See each package's distribution metadata for
its license and transitive dependencies.
