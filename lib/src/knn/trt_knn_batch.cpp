#include "knn/trt_knn_batch.hpp"

#include "common/trt_serialize.hpp"
#include "knn/trt_knn_batch_kernel.hpp"

#include <cstring>

namespace trt_mtr
{
namespace
{
static const char * PLUGIN_VERSION{"1"};
static const char * PLUGIN_NAME{"TRTKnnBatch"};
}  // namespace

KnnBatch::KnnBatch(const std::string & name, const int32_t top_k)
: mTopK(top_k), TRTPluginBase(name)
{
}

KnnBatch::KnnBatch(const std::string & name, const void * data, size_t length) : TRTPluginBase(name)
{
  deserialize_value(&data, &length, &mTopK);
}

KnnBatch::~KnnBatch() TRT_NOEXCEPT
{
}

nvinfer1::IPluginV2DynamicExt * KnnBatch::clone() const TRT_NOEXCEPT
{
  auto * plugin = new KnnBatch(mLayerName, mTopK);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs KnnBatch::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
  nvinfer1::IExprBuilder & exprBuilder) TRT_NOEXCEPT
{
  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = exprBuilder.constant(mTopK);

  return ret;
}

bool KnnBatch::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT
{
  return ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         ioDesc[pos].type == (pos < 2 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kINT32);
}

void KnnBatch::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc * outDesc, int nbOutputs) TRT_NOEXCEPT
{
  // Validate input arguments
}

size_t KnnBatch::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inDesc, int nbInputs,
  const nvinfer1::PluginTensorDesc * outDesc, int nbOutputs) const TRT_NOEXCEPT
{
  // TODO
  return 0;
}

int KnnBatch::enqueue(
  const nvinfer1::PluginTensorDesc * inDesc, const nvinfer1::PluginTensorDesc * outDesc,
  const void * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) TRT_NOEXCEPT
{
  const int n = inDesc[0].dims.d[0];
  const int m = inDesc[1].dims.d[0];

  const void * xyz = inputs[0];
  const void * query_xyz = inputs[1];
  const void * batch_idxs = inputs[2];
  const void * query_batch_offsets = inputs[3];
  const void * top_k = inputs[4];

  void * output = outputs[0];

  switch (outDesc[0].type) {
    case nvinfer1::DataType::kFLOAT:
      KnnBatchLauncher(
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

void KnnBatch::attachToContext(
  cudnnContext * cudnnCtx, cublasContext * cublasCtx,
  nvinfer1::IGpuAllocator * gpuAllocator) TRT_NOEXCEPT
{
  // TODO
}

void KnnBatch::detachFromContext() TRT_NOEXCEPT
{
  // TODO
}

nvinfer1::DataType KnnBatch::getOutputDataType(
  int index, const nvinfer1::DataType * inTypes, int nbInputs) const TRT_NOEXCEPT
{
  return nvinfer1::DataType::kINT32;
}

const char * KnnBatch::getPluginType() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * KnnBatch::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

int KnnBatch::getNbOutputs() const TRT_NOEXCEPT
{
  return 1;
}

size_t KnnBatch::getSerializationSize() const TRT_NOEXCEPT
{
  return sizeof(mTopK);
}

void KnnBatch::serialize(void * buffer) const TRT_NOEXCEPT
{
  serialize_value(&buffer, mTopK);
}

/* ====================== creator ====================== */
KnnBatchCreator::KnnBatchCreator()
{
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("top_k", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * KnnBatchCreator::getPluginName() const TRT_NOEXCEPT
{
  return PLUGIN_NAME;
}

const char * KnnBatchCreator::getPluginVersion() const TRT_NOEXCEPT
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt * KnnBatchCreator::createPlugin(
  const char * name, const nvinfer1::PluginFieldCollection * fc) TRT_NOEXCEPT
{
  const nvinfer1::PluginField * fields = fc->fields;
  int32_t top_k;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char * attrName = fields[i].name;
    if (!strcmp(attrName, "top_k")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      top_k = *reinterpret_cast<const int32_t *>(fields[i].data);
    }
  }
  auto plugin = new KnnBatch(name, top_k);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2DynamicExt * KnnBatchCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT
{
  auto plugin = new KnnBatch(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

}  // namespace trt_mtr
