# Overview

newton-neural is part of Newton, a project of the Linux Foundation that aims to be governed in a transparent,
accessible way for the benefit of the community. All participation in this project is open and not bound to
corporate affiliation. Participants are all bound to the Linux Foundation
[Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

# General Guidelines and Legal

Please refer to [the contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md)
in the `newton-governance` repository for general information, project membership, and legal requirements for
making contributions to Newton projects.

## Corporate Contributions and EasyCLA

If you contribute as part of your employment, your organization must be set up in LFX EasyCLA for Newton and
complete a Corporate Contributor License Agreement (CCLA). Opening a pull request against an EasyCLA-enabled
Newton repository triggers the EasyCLA check. If authorization is required, the check displays the prompt and
link for starting or completing the appropriate CLA workflow.

If your organization is not listed, the EasyCLA workflow lets you add it and identify the representative
authorized to coordinate the CCLA signing process. See
[Contributor License Agreements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#contributor-license-agreements)
for the full process.

# Contributing to newton-neural

Contributions from the community are welcome. To avoid surprises and to raise the chance of a contribution
being merged, share your plan first by opening a GitHub Issue or starting a Discussion. Keep each pull request
focused on one feature or one bug fix, and write clear, concise commit messages.

## Questions, Discussions, Suggestions

* Ask and answer questions in [GitHub Discussions](https://github.com/newton-physics/newton-neural/discussions).
* Use [GitHub Issues](https://github.com/newton-physics/newton-neural/issues) for bugs and features, that is,
  for what ends in a change of code or documentation.

## Reporting a Bug

* Check the [GitHub Issues](https://github.com/newton-physics/newton-neural/issues) for an existing report.
* Open a new Issue with a short, descriptive title and a clear description.
* Name the Git hash or release version, and the hardware and environment, with the output of `nvidia-smi`.
* Add code or a test that reproduces the bug.

## Fixing a Bug or Adding a Feature

* Write the change with a unit test that fails without it.
* Add a changelog fragment for a user-facing change, as [changelog/README.md](changelog/README.md) describes.
* Open a pull request that describes the change and mentions related issues.
* Have a signed CLA on file (see
  [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a
  [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members).

## Development Setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
uv run -m newton_neural.tests
```

```bash
uvx pre-commit run -a
```

The code follows Newton's
[source code and public API guidelines](https://github.com/newton-physics/newton/blob/main/CODING_GUIDELINES.rst).
The public API is what the modules directly under `newton_neural/` export. Everything under
`newton_neural/_src/` is private. Every source file starts with the SPDX header that
[.licenserc.yaml](.licenserc.yaml) checks.

## Improving Performance

Speed is a first priority of this library. A change that claims a speedup comes with a benchmark under
`asv/benchmarks` or with the numbers of an existing one, measured before and after on the same machine.
