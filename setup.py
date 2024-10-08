from setuptools import setup, find_packages
import os
import subprocess

# Function to dynamically fetch the version from VCS (git in this case)
def get_version():
    try:
        # Get the output from git describe
        version = subprocess.check_output(["git", "describe", "--tags", "--long"]).strip().decode("utf-8")
        
        # Remove the leading 'v' (if present) and convert the version to a valid PEP 440 format
        # Example: v1.0.1-1-g68c95a7 -> 1.0.1+1.g68c95a7
        match = re.match(r"v?(\d+\.\d+\.\d+)(?:-(\d+)-g([0-9a-f]+))?", version)
        if match:
            base_version = match.group(1)  # 1.0.1
            if match.group(2):  # Handle non-tagged commits (e.g., 1 commit after tag)
                version = f"{base_version}+{match.group(2)}.g{match.group(3)}"  # 1.0.1+1.g68c95a7
            else:
                version = base_version  # If it's exactly a tagged version
        return version
    except Exception:
        return "0.0.1"  # Fallback if no version info is found

setup(
    name="pyspecter",
    version=get_version(),  # Dynamic versioning based on VCS
    description="Framework for defining, building, and evaluating generalized shape observables for collider physics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Rikab Gambhir",
    author_email="rikab@mit.edu",
    url="https://github.com/rikab/SPECTER",
    project_urls={
        "Documentation": "https://github.com/rikab/SPECTER",
        "Homepage": "https://github.com/rikab/SPECTER",
        "Issue Tracker": "https://github.com/rikab/SPECTER/issues",
        "Releases": "https://github.com/rikab/SPECTER/releases",
        "Source Code": "https://github.com/rikab/SPECTER",
    },
    packages=find_packages(where="src"),  # Assume the package is in the "src" directory
    package_dir={"": "src"},  # Maps the root package to src/
    python_requires=">=3.7",
    install_requires=[
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "scipy>=1.5.1",
        "matplotlib>=3.5.0",
        "numpy",  # scipy controls compatible numpy versions
        "rikabplotlib @ git+https://github.com/rikab/rikabplotlib",
        "particleloader @ git+https://github.com/rikab/ParticleLoader"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=["shape observables", "jet physics"],
    license="MIT",
    include_package_data=True,
    # Optional build hook equivalent, though not strictly necessary for this simple setup.py
    data_files=[
        ("", ["LICENSE", "README.md", "pyproject.toml"])
    ],
    zip_safe=False,
)
