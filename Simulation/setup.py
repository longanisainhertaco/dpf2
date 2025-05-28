# setup.py
from setuptools import setup

setup(
    name='dpf_ai',
    version='0.1.0',
    description='Dense Plasma Focus hybrid Fluid–PIC simulator',
    author='Your Name',
    author_email='you@example.com',
    py_modules=[
        'circuit',
        'pic_solver',
        'hybrid_controller',
        'collisions_model',
        'radiation_model',
        'diagnostics',
        'fluid_solver_high_order',
        'fluid_models',
        'dpf_simulator_full_backend',
        'dpf_simulation',
        'dpf_simulator_amrex_backend',
        'dpf_simulator_server',
        'machine',
        'models',
        'utils',
        # optional warpX wrapper
        'warpx_wrapper',
    ],
    install_requires=[
        'numpy>=1.26.0',
        'h5py>=3.7.0',
        'pybind11>=2.13.0',
        'amrex_solver>=0.1.0',
    ],
    entry_points={
        'console_scripts': [
            'dpf_simulation = dpf_simulation:main',
        ],
    },
)

