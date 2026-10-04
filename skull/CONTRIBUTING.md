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

Numerical code (anything with a hand-written backward pass) needs a gradient check against finite
differences — see `tests/gradcheck.cpp`. Check that the test actually fails when you break the
code (e.g. drop a term from the derivative).

Language changes (parser, interpreter) are also covered by `tests/fuzz.cpp`: mutation fuzzing of the
parser, randomly generated programs and a model test that runs list/text operations in Skull and in a
C++ model. If you add a language feature, extend the generator or the model there. Tests were checked
by deliberately breaking the code under test, which is a good habit for new tests too.

`experimental/` is **not** part of the build; see its README before touching it.

## Making a release

Prebuilt packages (Linux, macOS arm64, Windows; AVX2 and `-compat` variants) are built by
`.github/workflows/release.yml`:

1. Raise `SKULL_VERSION` in `src/version.h`, move the `[Unreleased]` notes of `CHANGELOG.md` under the new version.
2. Merge to the default branch, then `git tag v<version> && git push origin v<version>`.
3. The workflow builds every package, runs the full test suite and a smoke test of the *unpacked* package,
   and attaches the archives plus `SHA256SUMS.txt` to a new GitHub release. It fails if the tag does not
   match `SKULL_VERSION`.

"Run workflow" (manual) builds and tests the packages without publishing anything.

## Questions?

Open an issue and ask. No question is too basic.
