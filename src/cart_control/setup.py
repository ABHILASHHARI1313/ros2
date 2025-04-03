from setuptools import find_packages, setup

package_name = 'cart_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abhilash-ts434',
    maintainer_email='abhilashhari1313@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'get_angle = cart_control.get_angle:main',
            'mpc = cart_control.mpc:main',
            'force = cart_control.force:main',
        ],
    },
)
