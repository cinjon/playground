import sys
from setuptools import setup, find_packages

CURRENT_PYTHON = sys.version_info[:2]
MIN_PYTHON = (3, 6)

if CURRENT_PYTHON < MIN_PYTHON:
    sys.stderr.write("""
        ============================
        Unsupported Python Version
        ============================
        
        Python {}.{} is unsupported. Please use a version newer than Python {}.{}.
    """.format(*CURRENT_PYTHON, *MIN_PYTHON))
    sys.exit(1)

with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

with open('requirements_extra.txt', 'r') as f:
    extras_require = f.readlines()

with open('VERSION') as f:
    VERSION = f.read().strip()

setup(name='pommerman',
      version=VERSION,
      description='PlayGround: AI Research into Multi-Agent Learning',
      url='https://www.pommerman.com',
      author='Pommerman',
      author_email='support@pommerman.com',
      license='Apache 2.0',
      classifiers=[
          'Programming Language :: Python :: 3.6',
      ],
      packages=find_packages(),
      install_requires=install_requires,
      extras_require={
        'extras': extras_require # @TODO this might need refinement
      },
      entry_points={
        'console_scripts': [
            'pom_battle=pommerman.cli.run_battle:main',
            'pom_tf_battle=pommerman.cli.train_with_tensorforce:main',
        ],
      },
      zip_safe=False)
