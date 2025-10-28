# TELOS Makefile - Quick Commands

.PHONY: help install test validate clean

help:
@echo “TELOS Make Commands:”
@echo “  make install    - Install all dependencies”
@echo “  make test       - Run unit tests”
@echo “  make validate   - Run Internal Test 0”
@echo “  make clean      - Remove generated files”
@echo “  make check      - Verify installation”

install:
pip install -e .

install-dev:
pip install -e “.[dev,viz,docs]”

check:
@echo “Checking installation…”
@python -c “from telos_purpose.core import primacy_math; print(‘✓ Core modules OK’)”
@python -c “import mistralai; print(‘✓ Mistral client OK’)”
@python -c “import sentence_transformers; print(‘✓ Embeddings OK’)”
@echo “✓ All checks passed”

test:
pytest tests/ -v

validate:
@echo “Running Internal Test 0…”
python -m telos_purpose.validation.run_internal_test0
@echo “\nGenerating summary…”
python -m telos_purpose.validation.summarize_internal_test0

clean:
rm -rf validation_results/
rm -rf **pycache**/
rm -rf telos_purpose/**pycache**/
rm -rf telos_purpose/**/**pycache**/
rm -rf .pytest_cache/
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
@echo “✓ Cleaned generated files”