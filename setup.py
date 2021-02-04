from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='image_array_and_histogram',
    version='1.0.0',
    author='Rishi Raj Singh Chauhan',
    author_email='rishirschauhan@gmail.com',
    description='Helps in getting array and histogram of a greyscale image. Also, image from the array.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rishi-chauhan/my-packages.git',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
