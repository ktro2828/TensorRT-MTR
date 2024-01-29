#include "attention/trt_attn_weight_computation.hpp"

#include "attention/trt_attn_weight_computation_kernel.hpp"

namespace trt_mtr
{
namespace
{
static const char * PLUGIN_VERSION{"1"};
static const char * PLUGIN_NAME{"TRTAttentionWeightComputation"};
}  // namespace

AttentionWeightComputation::AttentionWeightComputation(const std::string & name)
: TRTPluginBase(name)
{
}

AttentionWeightComputation::AttentionWeightComputation(
  const std::string & name, const void * data, size_t length)
: TRTPluginBase(name)
{
}

AttentionWeightComputation::~AttentionWeightComputation() TRT_NOEXCEPT
{
}

nvinfer1::IPluginV2DynamicExt * AttentionWeightComputation::clone() const TRT_NOEXCEPT
{
  auto * plugin = new AttentionWeightComputation(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs AttentionWeightComputation::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
  nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT
{
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[3].d[0];
  ret.d[1] = inputs[3].d[1];
  ret.d[2] = inputs[5].d[1];

  return ret;
}

bool AttentionWeightComputation::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT
{
  return ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         ioDesc[pos].type == (pos < 4 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kFLOAT);
}

void AttentionWeightComputation::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * outDesc, int nbOutputs) TRT_NOEXCEPT
{
  // Validate input arguments
}

size_t AttentionWeightComputation::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::PluginTensorDesc * outDesc, int nbOutputs) const TRT_NOEXCEPT
{
  // TODO
  return 0;
}

int AttentionWeightComputation::enqueue(
  const nvinfer1::PluginTensorDesc * inDesc, const nvinfer1::PluginTensorDesc * outDesc,
  const void * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) TRT_NOEXCEPT
{
  // parse query_batch_cnt description
  int batch = inDesc[0].dims.d[0];

  // parse index_pair description
  int total_query_num = inDesc[3].dims.d[0];
  int local_size = inDesc[3].dims.d[1];

  // parse key_features description
  int total_key_num = inDesc[5].dims.d[0];
  int nhead = inDesc[5].dims.d[1];
  int hdim = inDesc[5].dims.d[2];

  const void * query_batch_cnt = inputs[0];
  const void * key_batch_cnt = inputs[1];
  const void * index_pair_batch = inputs[2];
  const void * index_pair = inputs[3];
  const void * query_features = inputs[4];
  const void * key_features = inputs[5];

  void * output = outputs[0];

  switch (outDesc[0].type) {
    case nvinfer1::DataType::kFLOAT:
      AttentionWeightComputationLauncher<float>(
        batch, total_query_num, local_size, total_key_num, nhead, hdim,
        reinterpret_cast<const int *>(query_batch_cnt),
        reinterpret_cast<const int *>(key_batch_cnt),
        reinterpret_cast<const int *>(index_pair_batch), reinterpret_cast<const int *>(index_pair),
        reinterpret_cast<const float *>(query_features),
        reinterpret_cast<const float *>(key_features), reinterpret_cast<float *>(output), stream);
      break;

    default:
      break;
  }

  return 0;
}

void AttentionWeightComputation::attachToContext(
  cudnnContext * cudnnCtx, cublasContext * cublasCtx,
  nvinfer1::IGpuAllocator * gpuAllocator) TRT_NOEXCEPT
{
  // TODO
}

void AttentionWeightComputation::detachFromContext() TRT_NOEXCEPT
{
  // TODO
}

nvinfer1::DataType AttentionWeightComputation::getOutputDataType(
  int index, const nvinfer1::DataType * inTypes, int nbInputs) const TRT_NOEXCEPT
{
  return inTypes[4];
}

const char * AttentionWeightComputation::getPluginType() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * AttentionWeightComputation::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

int AttentionWeightComputation::getNbOutputs() const TRT_NOEXCEPT
{
  return 1;
}

size_t AttentionWeightComputation::getSerializationSize() const TRT_NOEXCEPT
{
  return 0;
}

void AttentionWeightComputation::serialize(void * buffer) const TRT_NOEXCEPT
{
}

/* ====================== creator ====================== */
AttentionWeightComputationCreator::AttentionWeightComputationCreator()
{
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * AttentionWeightComputationCreator::getPluginName() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * AttentionWeightComputationCreator::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt * AttentionWeightComputationCreator::createPlugin(
  const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT
{
  auto plugin = new AttentionWeightComputation(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2DynamicExt * AttentionWeightComputationCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT
{
  auto plugin = new AttentionWeightComputation(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace trt_mtr