from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cooka_door_state_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Praveen',
    maintainer_email='milooandmike@gmail.com',
    description='Door-state collision zone manager for cooka kitchen robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'door_manager  = cooka_door_state_sim.door_manager:main',
            'door_pnp_node = cooka_door_state_sim.door_pnp_node:main',
        ],
    },
)
