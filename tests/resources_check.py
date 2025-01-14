import pkg_resources

packages = [
    "numpy",
    "matplotlib",
    "collections",
    "pandas",
    "scipy",
    "sklearn",
    "tkinter",
    "pandastable",
    "seaborn",
]

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Not installed")
    except Exception as e:
        print(f"Error checking {package}: {e}")