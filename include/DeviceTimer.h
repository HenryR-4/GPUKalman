#ifndef __devicetimer_hpp
#define __devicetimer_hpp

#include <memory>

class DeviceTimer {
public:
  DeviceTimer();

  void start();

  void stop();

  float get_elapsed_time_ms();

private:
  class Implementation;
  std::shared_ptr<Implementation> implementation_;
};

#endif
