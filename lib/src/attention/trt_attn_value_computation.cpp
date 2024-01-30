#include "attention/trt_attn_value_computation.hpp"

#include "attention/trt_attn_value_computation_kernel.hpp"

namespace trt_mtr
{
namespace
{
static const char * PLUGIN_VERSION{"1"};
static const char * PLUGIN_NAME{"TRTAttentionValueComputation"};
}  // namespace

AttentionValueComputation::AttentionValueComputation(const std::string & name) : TRTPluginBase(name)
{
}

AttentionValueComputation::AttentionValueComputation(
  const std::string & name, const void * data, size_t length)
: TRTPluginBase(name)
{
}

AttentionValueComputation::~AttentionValueComputation() TRT_NOEXCEPT
{
}

nvinfer1::IPluginV2DynamicExt * AttentionValueComputation::clone() const TRT_NOEXCEPT
{
  auto * plugin = new AttentionValueComputation(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs AttentionValueComputation::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
  nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT
{
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[3].d[0];
  ret.d[1] = inputs[5].d[1];
  ret.d[2] = inputs[5].d[2];

  return ret;
}

bool AttentionValueComputation::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT
{
  return ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         ioDesc[pos].type == (pos < 4 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kFLOAT);
}

void AttentionValueComputation::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * outDesc, int nbOutputs) TRT_NOEXCEPT
{
  // Validate input arguments
}

size_t AttentionValueComputation::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::PluginTensorDesc * outDesc, int nbOutputs) const TRT_NOEXCEPT
{
  // TODO
  return 0;
}

int AttentionValueComputation::enqueue(
  const nvinfer1::PluginTensorDesc * inDesc, const nvinfer1::PluginTensorDesc * outDesc,
  const void * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) TRT_NOEXCEPT
{
  // parse query_batch_cnt description
  const int32_t batch = inDesc[0].dims.d[0];

  // parse index_pair description
  const int32_t total_query_num = inDesc[3].dims.d[0];
  const int32_t local_size = inDesc[3].dims.d[1];

  // parse value_features description
  const int32_t total_value_num = inDesc[5].dims.d[0];
  const int32_t nhead = inDesc[5].dims.d[1];
  const int32_t hdim = inDesc[5].dims.d[2];

  const void * query_batch_cnt = inputs[0];
  const void * key_batch_cnt = inputs[1];
  const void * index_pair_batch = inputs[2];
  const void * index_pair = inputs[3];
  const void * attn_weight = inputs[4];
  const void * value_features = inputs[5];

  void * output = outputs[0];

  switch (outDesc[0].type) {
    case nvinfer1::DataType::kFLOAT:
      AttentionValueComputationLauncher<float>(
        batch, total_query_num, local_size, total_value_num, nhead, hdim,
        reinterpret_cast<const int *>(query_batch_cnt),
        reinterpret_cast<const int *>(key_batch_cnt),
        reinterpret_cast<const int *>(index_pair_batch), reinterpret_cast<const int *>(index_pair),
        reinterpret_cast<const float *>(attn_weight),
        reinterpret_cast<const float *>(value_features), reinterpret_cast<float *>(output), stream);
      break;

    default:
      break;
  }

  return 0;
}

void AttentionValueComputation::attachToContext(
  cudnnContext * cudnnCtx, cublasContext * cublasCtx,
  nvinfer1::IGpuAllocator * gpuAllocator) TRT_NOEXCEPT
{
  // TODO
}

void AttentionValueComputation::detachFromContext() TRT_NOEXCEPT
{
  // TODO
}

nvinfer1::DataType AttentionValueComputation::getOutputDataType(
  int index, const nvinfer1::DataType * inTypes, int nbInputs) const TRT_NOEXCEPT
{
  return inTypes[4];
}

const char * AttentionValueComputation::getPluginType() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * AttentionValueComputation::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

int AttentionValueComputation::getNbOutputs() const TRT_NOEXCEPT
{
  return 1;
}

size_t AttentionValueComputation::getSerializationSize() const TRT_NOEXCEPT
{
  return 0;
}

void AttentionValueComputation::serialize(void * buffer) const TRT_NOEXCEPT
{
}

/* ====================== creator ====================== */
AttentionValueComputationCreator::AttentionValueComputationCreator()
{
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * AttentionValueComputationCreator::getPluginName() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * AttentionValueComputationCreator::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt * AttentionValueComputationCreator::createPlugin(
  const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT
{
  auto plugin = new AttentionValueComputation(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2DynamicExt * AttentionValueComputationCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT
{
  auto plugin = new AttentionValueComputation(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace trt_mtr