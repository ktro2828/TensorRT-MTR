#ifndef TRT_ATTN_VALUE_COMPUTATION_HPP
#define TRT_ATTN_VALUE_COMPUTATION_HPP

#include "common/trt_plugin_base.hpp"

namespace trt_mtr
{
/**
 * @brief Attention value computation plugin.
 *
 * @param query_batch_cnt <int; [B]>
 * @param key_batch_cnt <int; [B]>
 * @param index_pair_batch <int; [Q]>
 * @param index_pair <int; [Q, L]>
 * @param attn_weight <float; [Q, L, H]>
 * @param value_features <float; [Q, H, D]>
 * @return outputs <float; [Q, H, D]>
 */
class AttentionValueComputation : public TRTPluginBase
{
public:
  AttentionValueComputation(const std::string & name);
  AttentionValueComputation(const std::string & name, const void * data, size_t length);
  ~AttentionValueComputation() TRT_NOEXCEPT override;

  /* IPluginV2DynamicExt methods */
  nvinfer1::IPluginV2DynamicExt * clone() const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT override;

  bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs,
    int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * inDesc, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * outDesc, int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inDesc, int nbInputs,
    const nvinfer1::PluginTensorDesc * outDesc, int nbOutputs) const TRT_NOEXCEPT override;

  int enqueue(
    const nvinfer1::PluginTensorDesc * inDesc, const nvinfer1::PluginTensorDesc * outDesc,
    const void * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) TRT_NOEXCEPT override;

  void attachToContext(
    cudnnContext * cudnnCtx, cublasContext * cublasCtx,
    nvinfer1::IGpuAllocator * gpuAllocator) TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

  /* IPluginV2Ext methods */
  nvinfer1::DataType getOutputDataType(
    int index, const nvinfer1::DataType * inTypes, int nbInputs) const TRT_NOEXCEPT override;

  /* IPluginV2 methods */
  const char * getPluginType() const TRT_NOEXCEPT override;
  const char * getPluginVersion() const TRT_NOEXCEPT override;
  int getNbOutputs() const TRT_NOEXCEPT override;
  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void * buffer) const TRT_NOEXCEPT override;
};  // class AttentionValueComputation

class AttentionValueComputationCreator : public TRTPluginCreatorBase
{
public:
  AttentionValueComputationCreator();

  const char * getPluginName() const TRT_NOEXCEPT override;

  const char * getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT override;
};  // class AttentionValueComputationCreator
}  // namespace trt_mtr
#endif  // TRT_ATTN_VALUE_COMPUTATION_HPP