
import setuptools

setuptools.setup(
    name='Proces_Vel_Monit',                    # package name
    version='0.1',                          # version
    description='Workflow for processing ambient seismic noise to estimate velocity changes including implimentation of the Bayesian least square approach for velocity changes measurements',      # short description
    url='https://github.com/NBelovezhets/Ambient-noise-toolkit',               # package URL
    install_requires=[ "numpy", "scipy", "matplotlib", 'obspy', 'numba', 'seaborn', 'csr', 'progress' ],                    # list of packages this package depends
                                            # on.
    packages=["PROCESSING", 'DV_V'],              # List of module names that installing
    )
