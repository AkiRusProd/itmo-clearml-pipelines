# clearml-ml-pipelines

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.
```
export PYTHONUNBUFFERED=1
clearml-agent daemon --queue default
```
or
```
export PYTHONUNBUFFERED=1
clearml-agent execute --id 1cc3a7310de049f1ba5d7f6331df7ea6
```

```
export $(grep -v '^#' .env | xargs) && clearml-agent daemon --queue default
```

```
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

<!-- clearml-agent execute --id aec7f28b782a4459ac8969de3aef262b --standalone -->


rm -rf ~/.clearml/venvs-cache
rm -rf ~/.clearml/venvs-builds*