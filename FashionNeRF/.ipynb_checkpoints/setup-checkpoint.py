from setuptools import setup,find_packages

with open('requirements.txt') as requirement_file:
    requirements = requirement_file.read().split()

setup(
name='FashionNeRF',
description="Synthesising Virtual Fashion Try-On with Neural Radiance Fields",
version="1.0.0",
author="Molefe"
)
