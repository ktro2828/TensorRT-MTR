#include "mtr/mtr.hpp"

namespace mtr
{
TrtMTR::TrtMTR(
  const std::string & model_path, const std::string & precision,
  const std::vector<std::string> target_labels, const BatchConfig & batch_config,
  const size_t max_workspace_size, const BuildConfig & build_config)
: target_labels_(target_labels)
{
  builder_ = std::make_unique<MTRBuilder>(
    model_path, precision, batch_config, max_workspace_size, build_config);
  builder_->setup();

  if (!builder_->isInitialized()) {
    return;
  }
}
}  // namespace mtr