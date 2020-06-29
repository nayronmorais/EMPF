from setuptools import find_packages, setup


with open('requirements.txt') as reqf:
    requirements = reqf.read().split('\n')

setup(
    name='empf',
    packages=find_packages(where='src/models'),
    package_dir={'': 'src/models'},
    install_requires=requirements,
    version='0.1.0',
    description='Evolving Monitoring Processes Framework',
    author='Nayron Morais',
    author_email='nayronmorais@gmail.com',
    url='https://github.com/nayronmorais/EMPF',
    license='MIT',
    classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Manufacturing',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Legal Industry',
            'Natural Language :: English',
            'Topic :: System :: Monitoring',
            'Environment :: Console',
            'Operating System :: Microsoft',
            'Operating System :: Unix',
            'Operating System :: POSIX :: Linux',
            'Operating System :: MacOS',
    ],
)
