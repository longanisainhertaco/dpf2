# Continuous Integration & Dependency Pinning

Establish continuous integration (CI) that runs unit tests on each pull request
and pin Python dependencies to reproducible versions.

## Expected Inputs
- CI configuration file (e.g., GitHub Actions workflow) specifying the test job.
- `requirements.txt` or `pyproject.toml` with explicit version pins.

## Expected Outputs
- Automated CI run triggered by pull requests showing all tests pass.
- Artifact or log proving dependency versions used in the run.

## Acceptance Thresholds
- All tests in the CI pipeline succeed.
- Dependency files contain explicit version numbers for all packages.

## Demonstration
Provide a recording or linked CI run that demonstrates the automated tests
succeed and lists the pinned dependencies. Reference this document in the pull
request that adds the CI configuration.
