from setuptools import setup
import os
from glob import glob

package_name = 'pnp_operation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Praveen',
    maintainer_email='milooandmike@gmail.com',
    description='Pick-and-place operations for the cooka robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pnp_node = pnp_operation.pnp_node:main',
            'spawn_payload = pnp_operation.spawn_payload:main',
            'monitor = pnp_operation.monitor:main',
        ],
    },
)
