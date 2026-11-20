# newton-neural Development Guidelines

newton-neural is a library of neural robot dynamics for Newton. Its models come in families, and `FlashNeRD` is
the first.

Follow Newton's [source code and public API guidelines](https://github.com/newton-physics/newton/blob/main/CODING_GUIDELINES.rst).
This file adds what is specific to this repository.

- Use `uv` for every command.
- Run `uvx pre-commit run -a` before committing.
- The public API is what the modules directly under `newton_neural/` export through `__all__`. Implementation
  lives under `newton_neural/_src/` and is private.
- Lower layers never import higher ones when the package runs. An import under `TYPE_CHECKING` may name a type
  of a higher layer. `uv run lint-imports` checks the layers declared in `pyproject.toml`.
- A model family, such as `FlashNeRD`, is a subclass of `NeRDBase`. The tools of the library (recorder, losses,
  stepper, evaluation, solvers) reach a model through the members of `NeRDBase` and never import a family. Code
  and text that hold for any model do not name a family. Code and text that are about one family say so.
- In prose a model is a neural robot dynamics model, and `newton.Model` is the Newton model. In code a model is
  named `nerd` and `model` is the Newton model. The word family is for models. Systems, contact inputs,
  backbones and predictions are the kinds of parts a model is composed of.
- A contact input senses and encodes contact and names no place in a network, so that a model of any family may
  take it. Where its code enters is a setting of the family that builds the network, as `contact_injections` is
  of `FlashNeRD`.
- Every module starts with a docstring that says why the module exists and how it fits into the library, in
  full sentences. A public module says when to reach for it.
- MuJoCo is imported inside functions, never when `newton_neural` is imported.
- Code that runs in a simulation step reads nothing back to the host and allocates nothing after its first
  call, so that the step can be recorded into a CUDA graph.
- The options that make a deployed model fast have the same names wherever a model is deployed, `precision`,
  `compile` and `use_cuda_graph`. Each is switched on its own, and a solver and a stepper say what they run.
  A step never changes a setting of the process for longer than it runs.
- Add a Towncrier fragment under `changelog/` for a user-facing change.

## Tests

```bash
uv run -m newton_neural.tests
uv run -m newton_neural.tests -k test_stepper      # one module
```

Tests use `unittest` and register one test per device with `add_function_test`.

## Benchmarks

```bash
uvx --with virtualenv asv run --launch-method spawn HEAD^!
```
