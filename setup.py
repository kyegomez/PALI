from setuptools import setup, find_packages

setup(
    name='pali-torch',
    packages=find_packages(exclude=[]),
    version='0.0.2',
    license='MIT',
    description='Pali - PyTorch',
    author='Kye Gomez',
    author_email='kye@apac.ai',
    long_description_content_type='text/markdown',
    url='https://github.com/kyegomez/Pali',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'optimizers',
        'Prompt Engineering'
    ],
    install_requires=[
        'transformers',
        'torch',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)