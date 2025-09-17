from . import quick, compiling


def load_tests(loader, standard_tests, pattern):
    standard_tests.addTests((loader.loadTestsFromModule(module, pattern=pattern) for module in (quick, compiling)))
    return standard_tests
