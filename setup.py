from setuptools import find_packages, setup, Extension
from torch.utils import cpp_extension
import glob
import os
import subprocess
__version__ = "v1.0"

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_sources = glob.glob(os.path.join(current_dir, 'csrc', 'core', '*.cu'))
cpp_sources = glob.glob(os.path.join(current_dir, 'csrc', 'op', '*.cpp'))
py11_sources = glob.glob(os.path.join(current_dir, 'csrc', 'py11', '*.cpp'))
sources = cuda_sources + cpp_sources + py11_sources

cuda_include_paths = cpp_extension.include_paths(cuda=True)
self_include_paths = [os.path.join(current_dir, 'csrc')]
include_paths = cuda_include_paths + self_include_paths


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

_, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
if int(bare_metal_major) == 11:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"


setup(
    name='EET',
    version=__version__,
    author="dingjingzhen",
    author_email="dingjingzhen@corp.netease.com,ligongzheng@corp.netease.com,zhaosida@corp.netease.com",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='EET',
            sources=sources,
            include_dirs=include_paths,
            extra_compile_args={'cxx': ['-g',
                                        # "-U NDEBUG",
                                        ],

                                'nvcc': ['-U__CUDA_NO_HALF_OPERATORS__', 
                                         '-U__CUDA_NO_HALF_CONVERSIONS__',
                                         '-U__CUDA_NO_HALF2_OPERATORS__']},
            define_macros=[('VERSION_INFO', __version__),
                        #    ('_DEBUG_MODE_', None),
                           ]
            )
        ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension}
    )
