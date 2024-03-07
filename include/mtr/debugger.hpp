#ifndef MTR__DEBUGGER_HPP_
#define MTR__DEBUGGER_HPP_

#include <chrono>
#include <iostream>
#include <string>

namespace mtr
{
/**
 * @brief A class to debug the operation time.
 */
class Debugger
{
public:
  /**
   * @brief Create a event to measure the processing time.
   */
  void createEvent()
  {
    start_ = std::chrono::system_clock::now();
    has_event_ = true;
  }

  /**
   * @brief Display elapsed processing time from the event was created.
   *
   * @param prefix The message prefix. Defaults to `""`.
   */
  void printElapsedTime(const std::string & prefix = "")
  {
    if (!has_event_) {
      std::cerr << "There is no event." << std::endl;
    } else {
      end_ = std::chrono::system_clock::now();
      const auto elapsed_time = std::chrono::duration<double, std::milli>(end_ - start_).count();
      std::cout << prefix << elapsed_time << " ms" << std::endl;
      has_event_ = false;
    }
  };

private:
  std::chrono::system_clock::time_point start_, end_;
  bool has_event_{false};
};
}  // namespace mtr
#endif  // MTR__DEBUGGER_HPP_