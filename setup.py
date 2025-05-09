from setuptools import setup, find_packages

setup(
    name='chatosaurus-api',
    version='0.1.0',
    description='API for Chatosaurus',
    author='The Dino Stack',
    license='MIT',
    packages=find_packages(include=['llm_providers']),
    install_requires=[
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'chatosaurus-api=main:main'
        ]
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)