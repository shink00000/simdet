from setuptools import setup, find_packages


def requirements(file_name):
    with open(file_name, 'r') as f:
        requires = f.read().splitlines()
    return requires


setup(
    name='simdet',
    version='1.0.0',
    description='SimDet - Simple Detection',
    author='rs',
    packages=find_packages(include=('simdet')),
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Natural Language :: Japanese',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    license='Apache License 2.0',
    install_requires=requirements('requirements.txt')
)
