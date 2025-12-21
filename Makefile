PYTHON = python
SPHINX_APIDOC = sphinx-apidoc
SPHINX_BUILD = sphinx-build
PYROOTDIR = ./python
PYDOCSRCDIR = ./python/docs/source
PYDOCOUTDIR = ./python/docs/build

.PHONY: all req lint type test dist doc clean

all: lint type test dist doc

req:
	$(PYTHON) -m pip install --quiet --upgrade pip
	$(PYTHON) -m pip install --quiet setuptools wheel build cython pytest mypy flake8 sphinx sphinx-rtd-theme myst-parser

lint:
	$(PYTHON) -m flake8 --doctests $(PYROOTDIR)

type:
	$(PYTHON) -m pip install --quiet --force-reinstall ./dist/*.whl
	$(PYTHON) -m mypy --strict $(PYROOTDIR)

test:
	$(PYTHON) -m pip install --quiet --force-reinstall ./dist/*.whl
	$(PYTHON) -m pytest $(PYROOTDIR) -vv --doctest-modules

dist:
	$(PYTHON) -m build

doc:
	$(PYTHON) -m pip install --quiet --force-reinstall ./dist/*.whl
	$(SPHINX_APIDOC) -T -f -o $(PYDOCSRCDIR)/apidoc $(PYROOTDIR)/src
	$(SPHINX_BUILD) -b html $(PYDOCSRCDIR) $(PYDOCOUTDIR)/html

clean:
	rm -rf $(PYDOCOUTDIR) $(PYROOTDIR)/src/softfloatpy.egg-info ./dist
