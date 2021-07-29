"""Setup siamese."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 07月 17日
# ***
# ************************************************************************************/
#

from setuptools import setup, Extension
from torch.utils import cpp_extension

version = "0.0.1"
package_name = "siamese"
module_name = "siamese_cpp"
setup(
    name=package_name,
    version=version,
    ext_modules=[cpp_extension.CppExtension(module_name, ["siamese.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
