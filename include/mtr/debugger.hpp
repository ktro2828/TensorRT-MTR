#ifndef MTR__DEBUGGER_HPP_
#define MTR__DEBUGGER_HPP_

#include <chrono>
#include <iostream>
#include <string>

namespace mtr
{
class Debugger
{
public:
  void createEvent()
  {
    start_ = std::chrono::system_clock::now();
    has_event_ = true;
  }

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