from setuptools import setup, find_packages

setup(
    name='adalora_bi_eq13',
    version='0.1.0',
    description='AdaLoRA variant using Equation 1.3 (grad x activation) for importance',
    packages=find_packages(),
    install_requires=['torch','transformers','datasets','tqdm','numpy'],
)
