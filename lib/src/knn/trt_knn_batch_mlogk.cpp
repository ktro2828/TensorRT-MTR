#include "knn/trt_knn_batch_mlogk.hpp"

#include "common/trt_serialize.hpp"
#include "knn/trt_knn_batch_mlogk_kernel.hpp"

#include <cstring>

namespace trt_mtr
{
namespace
{
static const char * PLUGIN_VERSION{"1"};
static const char * PLUGIN_NAME{"TRTKnnBatchMlogK"};
}  // namespace

KnnBatchMlogK::KnnBatchMlogK(const std::string & name, const int32_t top_k)
: TRTPluginBase(name), mTopK(top_k)
{
}

KnnBatchMlogK::KnnBatchMlogK(const std::string & name, const void * data, size_t length)
: TRTPluginBase(name)
{
  deserialize_value(&data, &length, &mTopK);
}

KnnBatchMlogK::~KnnBatchMlogK() TRT_NOEXCEPT
{
}

nvinfer1::IPluginV2DynamicExt * KnnBatchMlogK::clone() const TRT_NOEXCEPT
{
  auto * plugin = new KnnBatchMlogK(mLayerName, mTopK);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs KnnBatchMlogK::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
  nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT
{
  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = exprBuilder.constant(mTopK);

  return ret;
}

bool KnnBatchMlogK::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT
{
  return ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         ioDesc[pos].type == (pos < 2 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kINT32);
}

void KnnBatchMlogK::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * outDesc, int nbOutputs) TRT_NOEXCEPT
{
  // Validate input arguments
}

size_t KnnBatchMlogK::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::PluginTensorDesc * outDesc, int nbOutputs) const TRT_NOEXCEPT
{
  // TODO
  return 0;
}

int KnnBatchMlogK::enqueue(
  const nvinfer1::PluginTensorDesc * inDesc, const nvinfer1::PluginTensorDesc * outDesc,
  const void * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) TRT_NOEXCEPT
{
  const int32_t n = inDesc[0].dims.d[0];
  const int32_t m = inDesc[1].dims.d[0];

  const void * xyz = inputs[0];
  const void * query_xyz = inputs[1];
  const void * batch_idxs = inputs[2];
  const void * query_batch_offsets = inputs[3];

  void * output = outputs[0];

  switch (outDesc[0].type) {
    case nvinfer1::DataType::kFLOAT:
      KnnBatchMlogKLauncher<float>(
        n, m, mTopK, reinterpret_cast<const float *>(xyz),
        reinterpret_cast<const float *>(query_xyz), reinterpret_cast<const int *>(batch_idxs),
        reinterpret_cast<const int *>(query_batch_offsets), reinterpret_cast<int *>(output),
        stream);
      break;

    default:
      break;
  }

  return 0;
}

void KnnBatchMlogK::attachToContext(
  cudnnContext * cudnnCtx, cublasContext * cublasCtx,
  nvinfer1::IGpuAllocator * gpuAllocator) TRT_NOEXCEPT
{
  // TODO
}

void KnnBatchMlogK::detachFromContext() TRT_NOEXCEPT
{
  // TODO
}

nvinfer1::DataType KnnBatchMlogK::getOutputDataType(
  int index, const nvinfer1::DataType * inTypes, int nbInputs) const TRT_NOEXCEPT
{
  return nvinfer1::DataType::kINT32;
}

const char * KnnBatchMlogK::getPluginType() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * KnnBatchMlogK::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

int KnnBatchMlogK::getNbOutputs() const TRT_NOEXCEPT
{
  return 1;
}

size_t KnnBatchMlogK::getSerializationSize() const TRT_NOEXCEPT
{
  return sizeof(mTopK);
}

void KnnBatchMlogK::serialize(void * buffer) const TRT_NOEXCEPT
{
  serialize_value(&buffer, mTopK);
}

/* ====================== creator ====================== */
KnnBatchMlogKCreator::KnnBatchMlogKCreator()
{
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("top_k", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * KnnBatchMlogKCreator::getPluginName() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * KnnBatchMlogKCreator::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt * KnnBatchMlogKCreator::createPlugin(
  const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT
{
  const nvinfer1::PluginField * fields = fc->fields;
  int32_t top_k;

  for (int i = 0; i < fc->nbFields; ++i) {
    const char * attrName = fields[i].name;
    if (!strcmp(attrName, "top_k")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      top_k = *static_cast<const int32_t *>(fields[i].data);
    }
  }
  auto plugin = new KnnBatchMlogK(name, top_k);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2DynamicExt * KnnBatchMlogKCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT
{
  auto plugin = new KnnBatchMlogK(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
}  // namespace trt_mtr