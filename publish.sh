# Build distribution
python setup.py sdist bdist_wheel

# Check distribution
twine check dist/*
