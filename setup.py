import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='image_array_and_histogram-rrsc',
    version='0.0.1',
    author='Rishi Raj Singh Chauhan',
    author_email='rishirschauhan@gmail.com',
    description='Helps in getting array and histogram of a greyscale image. Also, image from the array.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: Mozilla Public License 2.0',
        'Operating System :: OS Independent',
    ],
)
