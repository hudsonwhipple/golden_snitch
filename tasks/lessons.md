# Golden Snitch - Lessons Learned

## Patterns & Rules
<!-- Updated after corrections from the user -->

- **`pkg_resources` / TensorBoard:** There is no PyPI package named `pkg_resources`; it comes from **setuptools**. If TensorBoard fails with `No module named 'pkg_resources'`, you likely have **setuptools ≥ 82**, which removed `pkg_resources`. Fix: `pip install "setuptools>=68,<82"` (project pins this in `pyproject.toml`).

## Technical Insights
<!-- Insights discovered during implementation -->
