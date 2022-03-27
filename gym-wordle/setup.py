from setuptools import setup

setup(name='gym_wordle',
      version='0.0.2',
      install_requires=['colorama>=0.4.4'],
      package_data={'gym_wordle': ['data/*.txt', 'envs/*']},
      packages=['gym_wordle'],
      )
