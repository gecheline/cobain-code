from setuptools import setup, find_packages

setup(name='cobain',
      version='0.1.1',
      description='COntact Binary Atmospheres with INterpolation',
      url='https://github.com/gecheline/cobain',
      author='Angela Kochoska',
      author_email='a.kochoska@gmail.com',
      license='MIT License',
      packages=find_packages(),
      package_data={'cobain': ['structure/*.csv']},
      zip_safe=False)
