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
build.bat
```

## Running tests

```
skull.exe examples\tensor_test.skull
skull.exe examples\train_demo.skull
skull.exe examples\format_test.skull
skull.exe examples\generate_demo.skull
```

All four should run without errors.

## Questions?

Open an issue and ask. No question is too basic.
