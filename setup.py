from setuptools import setup, find_packages

setup(name='gym_vuledge',
      version='0.0.1',
      packages=['gym_vuledge'],
      package_dir={'gym_vuledge': 'gym_vuledge'},
      package_data={'gym_vuledge': ['gym_vuledge/envs/data/*']},
      install_requires=['gym', 'networkx', 'simpy', 'numpy'] 
)