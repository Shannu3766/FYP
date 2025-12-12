from setuptools import setup, find_packages
setup(
    name="adaptive_lora",
    version="2.1.0",
    description="Adaptive LoRA (Option B + CSV Logging) - reinitializes LoRA each epoch and logs BI/ranks.",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13",
        "transformers>=4.30",
        "peft>=0.3.0",
        "datasets>=2.0",
        "tqdm",
        "numpy",
        "scikit-learn",
        "accelerate",
        "pandas"
    ],
)
