/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass_extensions/compute_occupancy.h"

#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/ft_gemm_configs.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#include "cutlass_extensions/gemm/kernel/fpA_intB_gemm_with_broadcast.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include "../cutlass_heuristic.h"
#include "fpA_intB_gemm.h"
#include "cuda_utils.h"

namespace fastertransformer {

template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages>
void generic_mixed_gemm_kernelLauncher(const T*          A,
                                       const WeightType* B,
                                       const T*          weight_scales,
                                       const T*          biases,
                                       T*                C,
                                       int               m,
                                       int               n,
                                       int               k,
				       int bias_stride,
                                       CutlassGemmConfig gemm_config,
                                       char*             workspace,
                                       size_t            workspace_bytes,
                                       cudaStream_t      stream,
                                       int*              occupancy = nullptr)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
                  "Specialized for half, float");

    static_assert(cutlass::platform::is_same<T, WeightType>::value
                      || cutlass::platform::is_same<WeightType, uint8_t>::value
                      || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
                  "");

    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
    using ElementType       = ElementType_;

    using CutlassWeightType_ = typename cutlass::platform::
        conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t, WeightType>::type;
    using CutlassWeightType = CutlassWeightType_;

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.
    using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
    using ElementAccumulator  = typename MixedGemmArchTraits::AccType;

    using EpilogueOp =
        typename Epilogue<ElementType, MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
        ElementType,
        cutlass::layout::RowMajor,
        MixedGemmArchTraits::ElementsPerAccessA,
        CutlassWeightType,
        typename MixedGemmArchTraits::LayoutB,
        MixedGemmArchTraits::ElementsPerAccessB,
        ElementType,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        arch,
        ThreadblockShape,
        WarpShape,
        typename MixedGemmArchTraits::InstructionShape,
        EpilogueOp,
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        true,
        typename MixedGemmArchTraits::Operator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<typename GemmKernel_::Mma,
                                                          typename GemmKernel_::Epilogue,
                                                          typename GemmKernel_::ThreadblockSwizzle,
                                                          arch,  // Ensure top level arch is used for dispatch
                                                          GemmKernel_::kSplitKSerial>;

    if (occupancy != nullptr) {
        *occupancy = compute_occupancy_for_kernel<GemmKernel>();
        return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

    const int ldb =
        cutlass::platform::is_same<cutlass::layout::RowMajor, typename MixedGemmArchTraits::LayoutB>::value ?
            n :
            k * GemmKernel::kInterleave;

    typename Gemm::Arguments args({m, n, k},
                                  {reinterpret_cast<ElementType*>(const_cast<T*>(A)), k},
                                  {reinterpret_cast<CutlassWeightType*>(const_cast<WeightType*>(B)), ldb},
                                  {reinterpret_cast<ElementType*>(const_cast<T*>(weight_scales)), 0},
				  // TODO: Support more general bias shape
                                  {reinterpret_cast<ElementType*>(const_cast<T*>(biases)), bias_stride},
                                  {reinterpret_cast<ElementType*>(C), n},
                                  gemm_config.split_k_factor,
                                  {ElementAccumulator(1.f), ElementAccumulator(0.f)});

    // This assertion is enabled because because for the column interleaved layout, K MUST be a multiple of
    // threadblockK. The reason for this is that the default pitchlinear iterators are used to handle walking over the
    // interleaved matrix. The way masking in handled in these do not map to the interleaved layout. We need to write
    // our own predicated iterator in order to relax this limitation.
    if (GemmKernel::kInterleave > 1
        && ((k % MixedGemmArchTraits::ThreadblockK)
            || ((k / gemm_config.split_k_factor) % MixedGemmArchTraits::ThreadblockK))) {
        throw std::runtime_error("Temp assertion: k must be multiple of threadblockK");
    }

    Gemm gemm;
    if (gemm.get_workspace_size(args) > workspace_bytes) {
        FT_LOG_WARNING(
            "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg = "fpA_intB cutlass kernel will fail for params. Error: "
                              + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[FT Error][fpA_intB Runner] " + err_msg);
    }
    // printf("workspace size: %d, workspace_bytes: %d\n", gemm.get_workspace_size(args), workspace_bytes);

    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to initialize cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[FT Error][fpA_intB Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[FT Error][fpA_intB Runner] " + err_msg);
    }
}

template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages,
         typename Enable = void>
struct dispatch_stages {
    static void dispatch(const T*          A,
                         const WeightType* B,
                         const T*          weight_scales,
                         const T*          biases,
                         T*                C,
                         int               m,
                         int               n,
                         int               k,
			 int bias_stride,
                         CutlassGemmConfig gemm_config,
                         char*             workspace,
                         size_t            workspace_bytes,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch "
                              + std::to_string(arch::kMinComputeCapability) + " with stages set to "
                              + std::to_string(Stages);
        throw std::runtime_error("[FT Error][dispatch_stages::dispatch] " + err_msg);
    }
};

template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape>
struct dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2> {
    static void dispatch(const T*          A,
                         const WeightType* B,
                         const T*          weight_scales,
                         const T*          biases,
                         T*                C,
                         int               m,
                         int               n,
                         int               k,
			 int bias_stride,
                         CutlassGemmConfig gemm_config,
                         char*             workspace,
                         size_t            workspace_bytes,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_mixed_gemm_kernelLauncher<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(
													    A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
    }
};

template<typename T,
         typename WeightType,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape,
         int Stages>
struct dispatch_stages<T,
                       WeightType,
                       cutlass::arch::Sm80,
                       EpilogueTag,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
    static void dispatch(const T*          A,
                         const WeightType* B,
                         const T*          weight_scales,
                         const T*          biases,
                         T*                C,
                         int               m,
                         int               n,
                         int               k,
			 int bias_stride,
                         CutlassGemmConfig gemm_config,
                         char*             workspace,
                         size_t            workspace_bytes,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_mixed_gemm_kernelLauncher<T,
                                          WeightType,
                                          cutlass::arch::Sm80,
                                          EpilogueTag,
                                          ThreadblockShape,
                                          WarpShape,
                                          Stages>(
	  A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
    }
};

template<typename T,
         typename WeightType,
         typename arch,
         typename EpilogueTag,
         typename ThreadblockShape,
         typename WarpShape>
void dispatch_gemm_config(const T*          A,
                          const WeightType* B,
                          const T*          weight_scales,
                          const T*          biases,
                          T*                C,
                          int               m,
                          int               n,
                          int               k,
   			  int bias_stride,
                          CutlassGemmConfig gemm_config,
                          char*             workspace,
                          size_t            workspace_bytes,
                          cudaStream_t      stream,
                          int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemm_config.stages) {
        case 2:
            using DispatcherStages2 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>;
            DispatcherStages2::dispatch(
   	        A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
            break;
        case 3:
            using DispatcherStages3 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>;
            DispatcherStages3::dispatch(
   	        A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
            break;
        case 4:
            using DispatcherStages4 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>;
            DispatcherStages4::dispatch(
	        A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
            break;
        default:
            std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
            throw std::runtime_error("[FT Error][dispatch_gemm_config] " + err_msg);
            break;
    }
}

template<typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_to_cutlass(const T*          A,
                              const WeightType* B,
                              const T*          weight_scales,
                              const T*          biases,
                              T*                C,
                              int               m,
                              int               n,
                              int               k,
               		      int bias_stride,
                              char*             workspace,
                              size_t            workspace_bytes,
                              CutlassGemmConfig gemm_config,
                              cudaStream_t      stream,
                              int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Note that SIMT configs are omitted here since they are not supported for fpA_intB.
    // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually perform the best
    // for mixed type gemms.
    switch (gemm_config.tile_config) {
        case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<32, 128, 64>,
                                 cutlass::gemm::GemmShape<32, 32, 64>>(
                A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
            break;
        case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<64, 128, 64>,
                                 cutlass::gemm::GemmShape<64, 32, 64>>(
                A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
	    break;
        case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
            dispatch_gemm_config<T,
                                 WeightType,
                                 arch,
                                 EpilogueTag,
                                 cutlass::gemm::GemmShape<128, 128, 64>,
                                 cutlass::gemm::GemmShape<128, 32, 64>>(
                A, B, weight_scales, biases, C, m, n, k, bias_stride, gemm_config, workspace, workspace_bytes, stream, occupancy);
            break;
        case CutlassTileConfig::Undefined:
            throw std::runtime_error("[FT Error][fpA_intB][dispatch_gemm_to_cutlass] gemm config undefined.");
            break;
        case CutlassTileConfig::ChooseWithHeuristic:
            throw std::runtime_error(
                "[FT Error][fpA_intB][dispatch_gemm_to_cutlass] gemm config should have already been set by heuristic.");
            break;
        default:
            throw std::runtime_error(
                "[FT Error][fpA_intB][dispatch_gemm_to_cutlass] Config is invalid for mixed type GEMM.");
            break;
    }
}

template<typename T, typename WeightType>
CutlassFpAIntBGemmRunner<T, WeightType>::CutlassFpAIntBGemmRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    sm_ = getSMVersion();
    check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template<typename T, typename WeightType>
CutlassFpAIntBGemmRunner<T, WeightType>::~CutlassFpAIntBGemmRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T, typename WeightType>
template<typename EpilogueTag>
void CutlassFpAIntBGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag>(const T*          A,
                                                                            const WeightType* B,
                                                                            const T*          weight_scales,
                                                                            const T*          biases,
                                                                            T*                C,
                                                                            int               m,
                                                                            int               n,
                                                                            int               k,
									    int bias_stride,
                                                                            CutlassGemmConfig gemm_config,
                                                                            char*             workspace_ptr,
                                                                            const size_t      workspace_bytes,
                                                                            cudaStream_t      stream,
                                                                            int*              occupancy)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (sm_ >= 70 && sm_ < 75) {
        dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm70, EpilogueTag>(
	    A, B, weight_scales, biases, C, m, n, k, bias_stride, workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
    } else if (sm_ >= 75 && sm_ < 80) {
        dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm75, EpilogueTag>(
	    A, B, weight_scales, biases, C, m, n, k, bias_stride, workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
    } else if (sm_ >= 80 && sm_ < 90) {
        dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(
	    A, B, weight_scales, biases, C, m, n, k, bias_stride, workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
    }
    else {
        throw std::runtime_error(
            "[FT Error][CutlassFpAIntBGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS mixed type GEMM");
    }
}

template<typename T, typename WeightType>
template<typename EpilogueTag>
void CutlassFpAIntBGemmRunner<T, WeightType>::run_gemm<EpilogueTag>(const T*          A,
                                                                    const WeightType* B,
                                                                    const T*          weight_scales,
                                                                    const T*          biases,
                                                                    T*                C,
                                                                    int               m,
                                                                    int               n,
                                                                    int               k,
								    int bias_stride,
                                                                    char*             workspace_ptr,
                                                                    const size_t      workspace_bytes,
                                                                    cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr bool          is_weight_only    = !std::is_same<T, WeightType>::value;
    std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(sm_, is_weight_only, false);
    // printf("sm_: %d, is_weight_only: %d, candidate_configs.size(): %d\n", sm_, is_weight_only, candidate_configs.size());
    std::vector<int>               occupancies(candidate_configs.size());

    for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
        dispatch_to_arch<EpilogueTag>(A,
                                      B,
                                      weight_scales,
                                      biases,
                                      C,
                                      m,
                                      n,
                                      k,
				      bias_stride,
                                      candidate_configs[ii],
                                      workspace_ptr,
                                      workspace_bytes,
                                      stream,
                                      &occupancies[ii]);
    }
    // Standard GEMM, so 1 "expert". We use the same function for MoE and regular FFN.
    static constexpr int num_experts   = 1;
    CutlassGemmConfig    chosen_config = estimate_best_config_from_occupancies(candidate_configs,
                                                                            occupancies,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            num_experts,
                                                                            split_k_limit,
                                                                            workspace_bytes,
                                                                            multi_processor_count_,
                                                                            is_weight_only);

    dispatch_to_arch<EpilogueTag>(
        A, B, weight_scales, biases, C, m, n, k, bias_stride, chosen_config, workspace_ptr, workspace_bytes, stream);
}

template<typename T, typename WeightType>
void CutlassFpAIntBGemmRunner<T, WeightType>::gemm_bias_act(const T*          A,
                                                            const WeightType* B,
                                                            const T*          weight_scales,
                                                            const T*          biases,
                                                            T*                C,
                                                            int               m,
                                                            int               n,
                                                            int               k,
							    int bias_stride,
                                                            ActivationType    activation_type,
                                                            char*             workspace_ptr,
                                                            const size_t      workspace_bytes,
                                                            cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (activation_type) {
        case ActivationType::Relu:
            run_gemm<EpilogueOpBiasReLU>(
   	        A, B, weight_scales, biases, C, m, n, k, bias_stride, workspace_ptr, workspace_bytes, stream);
            break;
        case ActivationType::Gelu:
            run_gemm<EpilogueOpBiasFtGelu>(
   	        A, B, weight_scales, biases, C, m, n, k, bias_stride, workspace_ptr, workspace_bytes, stream);
            break;
        case ActivationType::Silu:
            run_gemm<EpilogueOpBiasSilu>(
   	        A, B, weight_scales, biases, C, m, n, k, bias_stride, workspace_ptr, workspace_bytes, stream);
            break;
        case ActivationType::Identity:
   	    run_gemm<EpilogueOpBias>(A, B, weight_scales, biases, C, m, n, k, bias_stride, workspace_ptr, workspace_bytes, stream);
            break;
        case ActivationType::InvalidType:
            FT_CHECK_WITH_INFO(false, "Activation type for fpA_intB must be valid.");
            break;
        default: {
            if (isGatedActivation(activation_type)) {
                FT_CHECK_WITH_INFO(false, "Fused gated activations not supported");
            }
            else {
                FT_CHECK_WITH_INFO(false, "Invalid activation type.");
            }
        }
    }
}

template<typename T, typename WeightType>
void CutlassFpAIntBGemmRunner<T, WeightType>::gemm(const T*          A,
                                                   const WeightType* B,
                                                   const T*          weight_scales,
                                                   T*                C,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   char*             workspace_ptr,
                                                   const size_t      workspace_bytes,
                                                   cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    run_gemm<EpilogueOpNoBias>(A, B, weight_scales, nullptr, C, m, n, k, 0, workspace_ptr, workspace_bytes, stream);
}

template <typename T, typename WeightType, typename Arch,
          typename ThreadblockShape, typename WarpShape, typename EpilogueOp,
          int stages>
void dispatch_gemm_residual(const T *A, const WeightType *B,
                            const T *weight_scales, const T *biases,
                            const T *residual, T *C, int m, int n, int k,
                            char *workspace_ptr, const size_t workspace_bytes,
                            cudaStream_t stream) {
  using ElementType = typename cutlass::platform::conditional<
      cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
  using ElementOutput = ElementType;

  using MixedGemmArchTraits =
      cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, WeightType, Arch>;
  using ElementAccumulator = typename EpilogueOp::ElementAccumulator;

  using Swizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using InstructionShape = typename MixedGemmArchTraits::InstructionShape;

  using Epilogue = typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      ElementType, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessA, WeightType,
      typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessB, ElementType,
      cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, Arch, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, Swizzle, stages,
      typename MixedGemmArchTraits::Operator>::Epilogue;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
      ElementType, cutlass::layout::RowMajor,
      MixedGemmArchTraits::ElementsPerAccessA, WeightType,
      typename MixedGemmArchTraits::LayoutB,
      MixedGemmArchTraits::ElementsPerAccessB, ElementType,
      cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, Arch, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, Swizzle, stages, true,
      typename MixedGemmArchTraits::Operator>::GemmKernel;

  using GemmKernel = cutlass::gemm::kernel::GemmFpAIntBWithBroadcast<
      typename GemmKernel_::Mma, Epilogue,
      typename GemmKernel_::ThreadblockSwizzle, Arch>;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

  // TODO: Support batch
  const int batch_count = 1;
  const auto lda = k;
  const int ldb =
      cutlass::platform::is_same<cutlass::layout::RowMajor,
                                 typename MixedGemmArchTraits::LayoutB>::value
          ? n
          : k * GemmKernel::kInterleave;
  const int ldc = n;

  typename Gemm::Arguments args(
      {m, n, k}, batch_count,
      {ElementAccumulator(1.f), ElementAccumulator(1.f)}, A, B, weight_scales,
      residual, C, biases, nullptr, 0, 0, 0, 0, 0, 0, lda, ldb, ldc, ldc, 0, 0);

  if (GemmKernel::kInterleave > 1 &&
      ((k % MixedGemmArchTraits::ThreadblockK) ||
       (k % MixedGemmArchTraits::ThreadblockK))) {
    throw std::runtime_error(
        "Temp assertion: k must be multiple of threadblockK");
  }

  Gemm gemm;
  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg =
        "fpA_intB cutlass kernel will fail for params. Error: " +
        std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FT Error][fpA_intB Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args, workspace_ptr, stream);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to initialize cutlass fpA_intB gemm. Error: " +
        std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("[FT Error][fpA_intB Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass fpA_intB gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("[FT Error][fpA_intB Runner] " + err_msg);
  }
}

template <typename T, typename WeightType, typename Arch, typename EpilogueOp,
          int stages>
void dispatch_gemm_residual(CutlassTileConfig tile_config, const T *A,
                            const WeightType *B, const T *weight_scales,
                            const T *biases, const T *residual, T *C, int m,
                            int n, int k, char *workspace_ptr,
                            const size_t workspace_bytes, cudaStream_t stream) {
  if (tile_config == CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64) {
    dispatch_gemm_residual<
        T, WeightType, Arch, cutlass::gemm::GemmShape<32, 128, 64>,
        cutlass::gemm::GemmShape<32, 32, 64>, EpilogueOp, stages>(
        A, B, weight_scales, biases, residual, C, m, n, k, workspace_ptr,
        workspace_bytes, stream);
  } else if (tile_config ==
             CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64) {
    dispatch_gemm_residual<
        T, WeightType, Arch, cutlass::gemm::GemmShape<64, 128, 64>,
        cutlass::gemm::GemmShape<64, 32, 64>, EpilogueOp, stages>(
        A, B, weight_scales, biases, residual, C, m, n, k, workspace_ptr,
        workspace_bytes, stream);
  } else { // CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
    dispatch_gemm_residual<
        T, WeightType, Arch, cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<128, 32, 64>, EpilogueOp, stages>(
        A, B, weight_scales, biases, residual, C, m, n, k, workspace_ptr,
        workspace_bytes, stream);
  }
}

template <typename T, typename WeightType, typename Arch, typename EpilogueOp>
void dispatch_gemm_residual(CutlassGemmConfig config, const T *A,
                            const WeightType *B, const T *weight_scales,
                            const T *biases, const T *residual, T *C, int m,
                            int n, int k, char *workspace_ptr,
                            const size_t workspace_bytes, cudaStream_t stream) {
  if constexpr (std::is_same<Arch, cutlass::arch::Sm75>::value) {
    dispatch_gemm_residual<T, WeightType, cutlass::arch::Sm75, EpilogueOp, 2>(
        config.tile_config, A, B, weight_scales, biases, residual, C, m, n, k,
        workspace_ptr, workspace_bytes, stream);
  } else if constexpr (std::is_same<Arch, cutlass::arch::Sm70>::value) {
    dispatch_gemm_residual<T, WeightType, cutlass::arch::Sm70, EpilogueOp, 2>(
        config.tile_config, A, B, weight_scales, biases, residual, C, m, n, k,
        workspace_ptr, workspace_bytes, stream);
  } else {
    if (config.stages == 3) {
      dispatch_gemm_residual<T, WeightType, Arch, EpilogueOp, 3>(
          config.tile_config, A, B, weight_scales, biases, residual, C, m, n, k,
          workspace_ptr, workspace_bytes, stream);
    } else if (config.stages == 4) {
      dispatch_gemm_residual<T, WeightType, Arch, EpilogueOp, 4>(
          config.tile_config, A, B, weight_scales, biases, residual, C, m, n, k,
          workspace_ptr, workspace_bytes, stream);
    } else { // 2
      dispatch_gemm_residual<T, WeightType, Arch, EpilogueOp, 2>(
          config.tile_config, A, B, weight_scales, biases, residual, C, m, n, k,
          workspace_ptr, workspace_bytes, stream);
    }
  }
}

template <typename T, typename WeightType, typename Arch,
          template <typename T_> class ActivationOp,
          template <typename T_> class BinaryOp>
inline void
dispatch_gemm_residual(CutlassGemmConfig config, const T *A,
                       const WeightType *B, const T *weight_scales,
                       const T *biases, const T *residual, T *C, int m, int n,
                       int k, const std::string &unary_op, char *workspace_ptr,
                       const size_t workspace_bytes, cudaStream_t stream) {
  using ElementOutput = T;
  using MixedGemmArchTraits =
      cutlass::gemm::kernel::MixedGemmArchTraits<T, WeightType, Arch>;
  using ElementAccumulator = typename MixedGemmArchTraits::AccType;

  if (unary_op == "identity") {
    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombinationResidualBlock<
            ElementOutput, ElementAccumulator, ElementAccumulator,
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ActivationOp, BinaryOp, cutlass::epilogue::thread::Identity>;
    dispatch_gemm_residual<T, WeightType, Arch, EpilogueOp>(
        config, A, B, weight_scales, biases, residual, C, m, n, k,
        workspace_ptr, workspace_bytes, stream);
  } else if (unary_op == "relu") {
    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombinationResidualBlock<
            ElementOutput, ElementAccumulator, ElementAccumulator,
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ActivationOp, BinaryOp, cutlass::epilogue::thread::ReLu>;
    dispatch_gemm_residual<T, WeightType, Arch, EpilogueOp>(
        config, A, B, weight_scales, biases, residual, C, m, n, k,
        workspace_ptr, workspace_bytes, stream);
  } else {
    throw std::runtime_error(
        "[FT Error][Unsupported unary op after residual block] " + unary_op);
  }
}

template <typename T, typename WeightType, typename Arch,
          template <typename T_> class ActivationOp>
void dispatch_gemm_residual(CutlassGemmConfig config, const T *A,
                            const WeightType *B, const T *weight_scales,
                            const T *biases, const T *residual, T *C, int m,
                            int n, int k, const std::string &binary_op,
                            const std::string &unary_op, char *workspace_ptr,
                            const size_t workspace_bytes, cudaStream_t stream) {
  if (binary_op == "plus") {
    dispatch_gemm_residual<T, WeightType, Arch, ActivationOp, cutlass::plus>(
        config, A, B, weight_scales, biases, residual, C, m, n, k, unary_op,
        workspace_ptr, workspace_bytes, stream);
  } else if (binary_op == "multiply") {
    dispatch_gemm_residual<T, WeightType, Arch, ActivationOp,
                           cutlass::multiplies>(
        config, A, B, weight_scales, biases, residual, C, m, n, k, unary_op,
        workspace_ptr, workspace_bytes, stream);
  } else {
    throw std::runtime_error(
        "[FT Error][Unsupported binary op for residual block] " + binary_op);
  }
}

template <typename T, typename WeightType, typename Arch>
void dispatch_gemm_residual(CutlassGemmConfig config, const T *A,
                            const WeightType *B, const T *weight_scales,
                            const T *biases, const T *residual, T *C, int m,
                            int n, int k, const std::string &activation,
                            const std::string &binary_op,
                            const std::string &unary_op, char *workspace_ptr,
                            const size_t workspace_bytes, cudaStream_t stream) {
  if (activation == "identity") {
    dispatch_gemm_residual<T, WeightType, Arch,
                           cutlass::epilogue::thread::Identity>(
        config, A, B, weight_scales, biases, residual, C, m, n, k, binary_op,
        unary_op, workspace_ptr, workspace_bytes, stream);
  } else if ("silu") {
    dispatch_gemm_residual<T, WeightType, Arch,
                           cutlass::epilogue::thread::SiLu>(
        config, A, B, weight_scales, biases, residual, C, m, n, k, binary_op,
        unary_op, workspace_ptr, workspace_bytes, stream);
  } else if ("relu") {
    dispatch_gemm_residual<T, WeightType, Arch,
                           cutlass::epilogue::thread::ReLu>(
        config, A, B, weight_scales, biases, residual, C, m, n, k, binary_op,
        unary_op, workspace_ptr, workspace_bytes, stream);
  } else if ("gelu") {
    dispatch_gemm_residual<T, WeightType, Arch,
                           cutlass::epilogue::thread::GELU>(
        config, A, B, weight_scales, biases, residual, C, m, n, k, binary_op,
        unary_op, workspace_ptr, workspace_bytes, stream);
  } else {
    throw std::runtime_error(
        "[FT Error][Unsupported activation before residual binary op] " +
        activation);
  }
}

template <typename T, typename WeightType>
void CutlassFpAIntBGemmRunner<T, WeightType>::gemm_bias_act_residual(
    const T *A, const WeightType *B, const T *weight_scales, const T *biases,
    const T *residual, T *C, int m, int n, int k, const std::string &activation,
    const std::string &binary_op, const std::string &unary_op,
    char *workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) {

  std::vector<CutlassGemmConfig> candidate_configs =
      get_candidate_configs(sm_, true, false);
  std::vector<int> occupancies(candidate_configs.size());

  for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
    dispatch_to_arch<EpilogueOpNoBias>(
        A, B, weight_scales, biases, C, m, n, k, 0, candidate_configs[ii],
        workspace_ptr, workspace_bytes, stream, &occupancies[ii]);
  }

  CutlassGemmConfig chosen_config = estimate_best_config_from_occupancies(
      candidate_configs, occupancies, m, n, k, 1, split_k_limit,
      workspace_bytes, multi_processor_count_, true);

  if (sm_ >= 80 && sm_ < 90) {
    dispatch_gemm_residual<T, WeightType, cutlass::arch::Sm80>(
        chosen_config, A, B, weight_scales, biases, residual, C, m, n, k,
        activation, binary_op, unary_op, workspace_ptr, workspace_bytes,
        stream);
  } else if (sm_ >= 75 && sm_ < 80) {
    dispatch_gemm_residual<T, WeightType, cutlass::arch::Sm75>(
        chosen_config, A, B, weight_scales, biases, residual, C, m, n, k,
        activation, binary_op, unary_op, workspace_ptr, workspace_bytes,
        stream);
  } else if (sm_ == 70) {
    dispatch_gemm_residual<T, WeightType, cutlass::arch::Sm70>(
        chosen_config, A, B, weight_scales, biases, residual, C, m, n, k,
        activation, binary_op, unary_op, workspace_ptr, workspace_bytes,
        stream);
  } else {
    throw std::runtime_error("[FT Error][Unsupported SM] " + sm_);
  }
}

template<typename T, typename WeightType>
int CutlassFpAIntBGemmRunner<T, WeightType>::getWorkspaceSize(const int m, const int n, const int k)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // TODO(masahi): Shouldn't it be 0?

    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int max_grid_m = (m + 31) / 32;
    const int max_grid_n = (n + 127) / 128;
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return max_grid_m * max_grid_n * split_k_limit * 4;
}

}  // namespace fastertransformer
