# Code Style

We use `Ruff==0.6` for both lint and format and `mypy==1.11.2`.

In root folder, install development required package in `dev_tools.dev-requirements.txt`.

```
pip install -r ./dev_tools/dev-requirements.txt
```

If using VScode, see more in [VScode](#vscode).

## Command Line

### Ruff

If not here are some helpful CLI command for code formatting and linting. See more usage in [Ruff](https://docs.astral.sh/ruff/)

```
ruff check                    # Lint all files in the current directory.
ruff check --fix              # Lint all files in the current directory, and fix any fixable errors.
ruff check --watch            # Lint all files in the current directory, and re-lint on change.
ruff check path/to/code/      # Lint all files in `path/to/code` (and any subdirectories).
ruff format                   # Format all files in the current directory.
ruff format path/to/code/     # Format all files in `path/to/code` (and any subdirectories).
ruff format path/to/file.py   # Format a single file.
```

### Ruff

See more in [Mypy](https://mypy.readthedocs.io/en/stable/config_file.html#disallow-dynamic-typing)

```
mypy .                        # check all files in this repo
```

## VScode

### Recommended VScode plugins

1. `Python` from Microsoft
2. `Ruff` from Astral Software
3. `Mypy` from Matan Gover

### Recommended settings for VScode

Use the created conda env for this repo in VScode. This is needed for mypy to function.

This is the recommended VScode configuration according to our code style, repo structure, and tools.
Please add `settings.json` file to `.vscode/` folder. We use this to ensure the local python and ruff version is the same as github CI's ruff version.

NOTE: Replace `/path/to/your/conda-venv` with the actual path to your conda virtual env.

Alternatively, you can get the path to python interpreter and ruff by running inside the conda venv:

```
conda activate decomfl
which python # For windows try: where python
which ruff   # For windows try: where ruff
```

```settings.json
{
	"editor.rulers": [100],
	"editor.detectIndentation": true,
	"[python]": {
		"editor.formatOnPaste": false,
		"editor.formatOnType": false,
		"editor.formatOnSave": true,
		"editor.defaultFormatter": "charliermarsh.ruff",
	},
	"ruff.enable": true,
	"ruff.interpreter": ["/path/to/your/conda-venv/python"], # or ["/path/to/your/conda-venv/python.exe"] for Windows
	"ruff.path": ["/path/to/your/conda-venv/bin/ruff"], # or ["/path/to/your/conda-venv/Scripts/ruff.exe"] for Windows
}
```
