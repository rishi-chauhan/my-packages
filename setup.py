from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='image_array_and_histogram',
    version='1.1.1',
    author='Rishi Raj Singh Chauhan',
    author_email='',
    description='Utilities to get arrays & histograms from grayscale images and build images from arrays.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rishi-chauhan/my-packages.git',
    project_urls={
        'Issue Tracker': 'https://github.com/rishi-chauhan/my-packages/issues',
    },
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'Pillow>=9.0',
    ],
    extras_require={
        'dev': ['pytest>=7']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    keywords='image histogram numpy pillow grayscale',
    license='MIT',
)
