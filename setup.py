from setuptools import find_packages, setup, Extension
from torch.utils import cpp_extension
import glob
import os

__version__ = "0.0.1 beta"

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_sources = glob.glob(os.path.join(current_dir, 'csrc', 'core', '*.cu'))
cpp_sources = glob.glob(os.path.join(current_dir, 'csrc', 'op', '*.cpp'))
py11_sources = glob.glob(os.path.join(current_dir, 'csrc', 'py11', '*.cpp'))
sources = cuda_sources + cpp_sources + py11_sources

cuda_include_paths = cpp_extension.include_paths(cuda=True)
self_include_paths = [os.path.join(current_dir, 'csrc')]
include_paths = cuda_include_paths + self_include_paths

setup(
    name='EET',
    version=__version__,
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='EET',
            sources=sources,
            include_dirs=include_paths,
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-U__CUDA_NO_HALF_OPERATORS__', 
                                         '-U__CUDA_NO_HALF_CONVERSIONS__',
                                         '-U__CUDA_NO_HALF2_OPERATORS__']},
            define_macros=[('VERSION_INFO', __version__)]
            )
        ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension}
    )
