from setuptools import find_packages, setup, Extension
from torch.utils import cpp_extension
from glob import glob
import os
import subprocess
__version__ = "2.0.0-alpha.0"

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_sources = glob(os.path.join(current_dir, 'csrc', 'core', '*.cu'))
cpp_sources = glob(os.path.join(current_dir, 'csrc', 'op', '*.cpp'))
py11_sources = glob(os.path.join(current_dir, 'csrc', 'py11', '*.cpp'))
cutlass_sources = glob(os.path.join(current_dir, '3rdparty', 'cutlass_kernels', '*.cu')) + \
                  glob(os.path.join(current_dir, '3rdparty', 'cutlass_kernels', '*.cc')) + \
                  glob(os.path.join(current_dir, '3rdparty', 'utils', '*.cc'))

sources = cuda_sources + cpp_sources + py11_sources + cutlass_sources

include_paths = []
include_paths.append(cpp_extension.include_paths(cuda=True))    # cuda path
include_paths.append(os.path.join(current_dir, 'csrc'))         # csrc path
include_paths.append(os.path.join(current_dir, '3rdparty'))
include_paths.append(os.path.join(current_dir, '3rdparty/utils'))
include_paths.append(os.path.join(current_dir, '3rdparty/cutlass_kernels'))
include_paths.append(os.path.join(current_dir, '3rdparty/cutlass/include'))
include_paths.append(os.path.join(current_dir, '3rdparty/cutlass_extensions/include'))



def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


_, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
if int(bare_metal_major) >= 11:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6+PTX"
else:
    raise RuntimeError(
        "EET is only supported on CUDA 11 and above.  "
        "Your version of CUDA is: {}\n".format(bare_metal_major)
    )


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
                                        '-std=c++17',
                                        # "-U NDEBUG",
                                        ],

                                'nvcc': ['-std=c++17',
                                         '-U__CUDA_NO_HALF_OPERATORS__',
                                         '-U__CUDA_NO_HALF_CONVERSIONS__',
                                         '-U__CUDA_NO_HALF2_OPERATORS__']},
            define_macros=[
                ('VERSION_INFO', __version__),
                # ('_DEBUG_MODE_', None),
            ]
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension}
)
