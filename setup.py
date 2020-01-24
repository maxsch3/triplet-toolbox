from setuptools import setup
from setuptools import find_packages

long_description = '''
A set of tools for implementing triplet networks in tensorflow/keras. The toolbox includes a set 
of loss functions that plug in to tensorflow/keras neural network seamlessly, transforming
your model into a one-short learning triplet model  
'''

setup(name='triplet-tools',
      version='0.2.0',
      description='A toolbox for creating and training triplet networks in tensorflow',
      long_description=long_description,
      author='Maxim Scherbak',
      author_email='maxim.scherbak@gmail.com',
      url='https://github.com/maxsch3/triplet-toolbox',
      download_url='https://github.com/maxsch3/triplet-toolbox',
      license='MIT',
      install_requires=['tensorflow>=1.14.0'],
      extras_require={
          'tests': ['pytest',
                    'markdown'],
      },
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())