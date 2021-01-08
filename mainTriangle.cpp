#include "src/MainVulkan.h"
#include <cstdlib>
#include <iostream>

int main() {
  MainVulkan* m = new MainVulkan();
  try {
    m->run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;

  }
return EXIT_SUCCESS;
}
