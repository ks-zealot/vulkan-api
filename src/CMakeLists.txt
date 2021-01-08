cmake_minimum_required(VERSION 2.8)	 # Проверка версии CMake.
										# Если версия установленой программы
										# старее указаной, произайдёт аварийный выход.
set(CMAKE_CXX_STANDARD 17)
project(mainvulkan)			# Название проекта
find_package(glfw3 3.2.1 REQUIRED)
find_package(Vulkan REQUIRED FATAL_ERROR)


add_library(mainvulkan MainVulkan.cpp MainVulkan.h)	# Создает исполняемый файл с именем main


target_link_libraries(mainvulkan vulkan  glfw)		# Линковка программы с библиотекой
