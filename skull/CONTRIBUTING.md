# Contributing to Skull

First of all — thank you for wanting to help. Skull is built on the idea that AI training should be accessible to everyone, and every contribution moves that forward.

## What we need help with

- **Bug reports** — if something breaks, open an issue
- **New examples** — `.skull` programs that show what Skull can do
- **Performance improvements** — faster math, better SIMD usage
- **Documentation** — explaining how things work
- **Testing** — running Skull on different hardware and reporting results
- **Ideas** — what should Skull be able to do that it can't yet?

## How to contribute

1. Fork the repository
2. Create a branch: `git checkout -b my-feature`
3. Make your changes
4. Test that it still builds and runs
5. Open a pull request

## Code style

- C++17
- Headers only (`.h` files) — no separate `.cpp` except `main.cpp`
- Comments in English
- Error messages that actually explain what went wrong

## Building

```
cd skull
./build.sh        # Linux / macOS
build.bat         # Windows (Visual Studio 2019+ and CMake)
```

Or directly with CMake: `cmake -S . -B build && cmake --build build`.

## Running tests

```
./build.sh --test         # build + run everything   (Windows: build.bat --test)
cd build && ctest         # run the tests again
```

Each test runs a `.skull` script and checks exit code and output (see `tests/CMakeLists.txt`).
To add one, drop a script into `tests/cases/` and register it with `skull_case(...)`.
Please add a regression test with every bug fix.

Memory errors: build with `-DSKULL_SANITIZE=ON` (gcc/clang) and run `ctest`; this enables
AddressSanitizer, LeakSanitizer and UBSan. CI does this on every push.

`experimental/` is **not** part of the build; see its README before touching it.

## Questions?

Open an issue and ask. No question is too basic.
