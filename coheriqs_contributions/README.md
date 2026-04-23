# My Contributions

This folder contains all custom contributions and new files created for the MOABB benchmarking pipeline project.

## Structure

- **pipelines/** — YAML pipeline configuration files
  - `MyCustomPipeline.yml` — Template for a custom pipeline definition

- **moabb_pipelines/** — Custom Python implementations
  - `custom_classifiers.py` — Custom classifier implementations and components

## Usage

To integrate these files into the main project:

1. Copy files from `pipelines/` to `/pipelines/`
2. Copy files from `moabb_pipelines/` to `/moabb/pipelines/`
3. Update any imports in the main `__init__.py` files as needed

## Notes

- All YAML files follow the MOABB pipeline configuration format
- All Python files implement scikit-learn compatible interfaces (BaseEstimator, ClassifierMixin, etc.)
- Customize the templates by editing the pipeline definitions and classifier implementations
