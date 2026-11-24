# Changelog fragments

Pull requests with user-facing changes add a changelog fragment to this folder instead of editing
`CHANGELOG.md`. Towncrier combines the fragments when a release is cut. The conventions are the ones of
[Newton](https://github.com/newton-physics/newton/blob/main/changelog/README.md).

Name a fragment after the GitHub issue it belongs to and its category.

```text
128.added.md
128.fixed.md
```

Without an issue, use a readable name with a short random suffix. The leading `+` tells Towncrier not to
create a link.

```text
+contact-set-capacity-a1b2c3d4.changed.md
```

The categories, in rendered order, are `added`, `changed`, `deprecated`, `removed` and `fixed`. Each file
is one entry. Write imperative present tense and end with a period. For `changed`, `deprecated` and
`removed`, say what a user has to do.

Preview the result with

```bash
uvx --from towncrier==25.8.0 towncrier build --draft --version X.Y.Z --date YYYY-MM-DD
```
