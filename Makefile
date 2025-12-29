ifneq ($(strip $(AUDITWHEEL)),)
WHEEL_OUTDIR = ./wheelhouse
else
WHEEL_OUTDIR = ./dist
endif
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
ifneq ($(strip $(AUDITWHEEL)),)
	$(PYTHON) -m pip install --quiet auditwheel
endif

lint:
	$(PYTHON) -m flake8 --doctests $(PYROOTDIR)

type:
	$(PYTHON) -m pip install --quiet --force-reinstall $(WHEEL_OUTDIR)/softfloatpy-*.whl
	$(PYTHON) -m mypy --strict $(PYROOTDIR)

test:
	$(PYTHON) -m pip install --quiet --force-reinstall $(WHEEL_OUTDIR)/softfloatpy-*.whl
	$(PYTHON) -m pytest $(PYROOTDIR)/tests -vv --doctest-modules

dist:
	$(PYTHON) -m build
ifneq ($(strip $(AUDITWHEEL)),)
	auditwheel repair dist/softfloatpy-*.whl
endif

doc:
	$(PYTHON) -m pip install --quiet --force-reinstall $(WHEEL_OUTDIR)/softfloatpy-*.whl
	$(SPHINX_APIDOC) -T -f -o $(PYDOCSRCDIR)/apidoc $(PYROOTDIR)/src
	$(SPHINX_BUILD) -b html $(PYDOCSRCDIR) $(PYDOCOUTDIR)/html

clean:
	rm -rf $(PYDOCOUTDIR) $(PYROOTDIR)/src/softfloatpy.egg-info ./dist ./wheelhouse $(find . -name '__pycache__')
