# educational C++ Reverse-Mode Autograd Engine

this is a small, offline, CPU-only C++20 implementation of scalar reverse-mode automatic differentiation. It implements the essential ideas behind PyTorch-style autograd without a machine-learning framework or any non-standard dependency. The included executable trains a tiny MLP on XOR.

## Build and run

Requires CMake 3.20+ and a C++20 compiler (Visual Studio 2022, clang, or GCC). All inputs are generated locally; build and runtime do not need Internet access.

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\autograd_demo.exe
```

For single-configuration generators, omit `-C Release` and run `./build/autograd_demo` (or the platform equivalent). The project has no external dependencies and is appropriate for a modest Windows x64 CPU. The default XOR run uses 1,200 scalar-gradient steps and completes quickly.

## Architecture and ownership

`Value` is one scalar graph node: `data`, accumulated `grad`, parent nodes, and a local backward closure. A result node owns its parents with `std::shared_ptr`, which means an expression result keeps all intermediates alive. Its closure holds a `std::weak_ptr` to itself, avoiding a reference cycle; parents are only referenced backward, so the graph remains a DAG with no dangling references.

`backward()` first creates a post-order topological list through a visited set. It seeds the scalar output derivative with one, then invokes local rules in reverse topological order. Every local rule uses `+=`, so paths that meet at a shared variable correctly accumulate. This is reverse-mode AD: one reverse pass computes derivatives of one scalar loss with respect to arbitrarily many parameters, generally much cheaper than separately differentiating the loss for each parameter.

Calling `backward()` (the default) clears all reachable gradients before computing a new derivative. Calling `backward(true)` retains leaf gradients and adds one new vector-Jacobian product; it clears intermediate gradients first, preventing stale intermediates from being propagated again. `zero_grad()` clears the node and every ancestor.

## Chain rule and local rules

For `z=f(x,y)`, reverse mode applies `x.grad += z.grad * dz/dx` and similarly for `y`. The implemented rules are:

| Operation | Forward | Local derivative(s) |
|---|---|---|
| addition/subtraction | `a ± b` | `da=1`, `db=±1` |
| multiplication | `a*b` | `da=b`, `db=a` |
| division | `a/b` | `da=1/b`, `db=-a/b²` |
| negation | `-a` | `-1` |
| scalar power | `a^c` | `c*a^(c-1)` |
| exponential | `exp(a)` | `exp(a)` |
| logarithm | `log(a)` | `1/a` |
| tanh | `tanh(a)` | `1-tanh(a)²` |
| ReLU | `max(0,a)` | `1` for `a>0`, otherwise `0` |
| sigmoid | `s(a)` | `s(a)*(1-s(a))` |

Domain behavior is deliberate: division by zero, `log(a)` for `a<=0`, zero to a negative power, and a fractional power of a negative base throw `std::domain_error`. `exp` throws if its forward result is non-finite. ReLU’s non-differentiable derivative at zero is conventionally defined as zero. Finite-difference checks avoid non-smooth boundaries.

## Expression API and MLP

Nodes are `ValuePtr`s and use readable operators, e.g. `auto loss = exp(x) * log(x) + pow(x, 3.0) / 2.0;`. `MLP` consists of C++-implemented affine neurons, `tanh` hidden units, a sigmoid output, binary cross-entropy, deterministic initialization, and vanilla SGD. The demo uses the generated XOR samples; for a local dataset, parse a local CSV/text file with standard C++ I/O and pass its feature rows to `MLP::operator()`. No runtime downloader is included or needed.

## Tests

`autograd_tests` covers shared/reused variables, branches, multiple inputs, constants, default and accumulating backwards, zeroing, invalid domains, and central-difference gradient checks. It uses

`(f(x+h)-f(x-h))/(2h)`, with `h=1e-6` and tolerance `2e-5`—a practical tolerance that covers floating-point roundoff and truncation error in `double` precision.

## Limitations and tensor path

This is intentionally scalar: each arithmetic result allocates one node, which is educational but not efficient for large workloads. Tensor generalization would require array storage with shapes/strides, broadcasting-aware local derivatives, reduction operations, batched matrix multiplication, and an efficient tensor-aware backward graph. It would also require carefully designed memory reuse and likely SIMD/BLAS acceleration; those are outside this dependency-free implementation.
