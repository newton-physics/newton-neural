# Security Policy

newton-neural is part of Newton, a Linux Foundation project that is built and maintained by its community.

## Scope

The reporting policy and handling commitments of every repository in the
[newton-physics organization](https://github.com/newton-physics) are defined in the
[newton-governance security policy](https://github.com/newton-physics/newton-governance/blob/main/SECURITY.md).
This document adds what is specific to newton-neural.

## Supported Versions

Only the most recent minor release line receives security fixes.

## Reporting a Vulnerability

Please **do not open a public GitHub issue, pull request or discussion** for a potential vulnerability.
Use the **Security** tab of the
[newton-neural repository](https://github.com/newton-physics/newton-neural/security) and select
**Report a vulnerability**. The report reaches the maintainers confidentially.

Include the version, branch or commit, the affected component, steps to reproduce, and the impact you expect.

A flaw in a dependency such as Newton, Warp, MuJoCo or PyTorch belongs to that project's own security process.
Report it here as well when the way newton-neural uses the dependency is what exposes users.

## What the Library Loads

newton-neural runs inside the process of the application that imports it. It is not a sandbox and not a
boundary for privileges or tenants.

- **Checkpoints.** A checkpoint is one file of tensors and plain data. It is read with
  `torch.load(weights_only=True)` and its parts are rebuilt by class name from registries, so loading a
  checkpoint imports no module and runs no code from the file.
- **Datasets.** A dataset folder holds NumPy arrays and one JSON file. Arrays are read without pickle.
- **Compiled code.** PyTorch and Warp compile and cache kernels on the machine that runs them. The caches
  are trusted like the rest of the local installation.

Models, datasets and scenes from sources you do not trust can still be wrong, large or slow. Applications
that accept them from users are responsible for limits and isolation.
