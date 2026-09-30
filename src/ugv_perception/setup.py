from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ugv_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tejasvpatel',
    maintainer_email='tejasvpatel@example.com',
    description='UGV perception system',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = ugv_perception.perception_node:main',
            'camera_node = ugv_perception.camera_node:main',
	    'segmentation_node = ugv_perception.segmentation_node:main',
       	    'yolop_node = ugv_perception.yolop_node:main',
	    'drivable_mask_node = ugv_perception.drivable_mask_node:main',
	    'steering_analysis_node = ugv_perception.steering_analysis_node:main',
            'obstacle_safety_node = ugv_perception.obstacle_safety_node:main',
	    'navigation_controller_node = ugv_perception.navigation_controller_node:main',
            'navigation_grid_node = ugv_perception.navigation_grid_node:main',
	    'astar_planner_node = ugv_perception.astar_planner_node:main',
	 ],
    },
)
