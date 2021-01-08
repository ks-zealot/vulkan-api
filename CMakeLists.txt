cmake_minimum_required(VERSION 2.8)	 # Проверка версии CMake.
										# Если версия установленой программы
SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++11")
SET (CMAKE_CXX_FLAGS                "-O2")

										# старее указаной, произайдёт аварийный выход.
set(CMAKE_CXX_STANDARD 17)
project(hello_world)			# Название проекта
add_subdirectory(src)

set(SOURCE_EXE mainTriangle.cpp)		# Установка переменной со списком исходников для исполняемого файла
execute_process (
        COMMAND /bin/bash compile_shaders.sh
	WORKING_DIRECTORY shaders
)
add_executable(VulkanApp ${SOURCE_EXE})	# Создает исполняемый файл с именем main

target_link_libraries(VulkanApp mainvulkan)		# Линковка программы с библиотекой
