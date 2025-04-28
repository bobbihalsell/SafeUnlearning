from setuptools import setup, find_packages

setup(
    name="safe_unlearning",
    version="0.1.0",
    package_dir={"": "src"},  # Points to 'src/' as the package root
    packages=find_packages(where="src"),  # Finds packages inside 'src/'
    install_requires=[],
)
