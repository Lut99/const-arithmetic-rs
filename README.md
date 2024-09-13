# const-arithmetic-rs
Implements procedural macro for doing math on integer literals for use in Rust macros.


## Installation
To use this crate in one of your projects, simply add it to your `Cargo.toml` file:
```toml
[dependencies]
literal-arithmetic = { git = "https://github.com/Lut99/literal-arithmetic-rs" }
```

You can also depend on a specific tag if you specify it so:
```toml
[dependencies]
literal-arithmetic = { git = "https://github.com/Lut99/literal-arithmetic-rs", tag = "v0.1.0" }
```


## Usage
The crate revolves around the `calc!(...)`-macro, which accepts (simple) integer expressions and expands to a literal with the target value.

For example:
```rust
use literal_arithmetic::calc;

assert_eq!(calc!(1 + 1), 2);
```

> Tip: use the [`cargo-expand`](https://github.com/dtolnay/cargo-expand)-subcommand to verify the `2` literal is being produced instead of the sum.

Only basic operations are supported. Specifically:
- Integer literals
    - This includes both plain literals (`1`) or typed literals (`1usize`). The resulting literal will carry over the type according to the following rules:
        - Any sum involving a plain and a typed literal will become typed;
        - If there are two typed, then the result will be the larger of the two and signed if one of the two is signed.
- Binary operators
    - Arithmetic operators (`+`, `-`, `*`, `/` or `%`); and
    - Bitwise operators (`&`, `|`, `^`, `>>` or `<<`).
- Unary operators
    - `-`
- Parenthesis
- Casts (as long as its to other integer types)

Some examples can be found in the procedural macro's documentation.


## Contributing
Any ideas, suggestions and fixes are welcome! Feel free to [raise an issue](https://github.com/Lut99/literal-arithmetic-rs/issues) or [create a pull request](https://github.com/Lut99/literal-arithmetic-rs/pulls).


## License
This project is licensed under the Apache 2.0 license. See [`LICENSE`](./LICENSE) for more information.
