# Only the writer-interop and chat-template script tests need an interpreter; the C++/CUDA
# unit tests must stay buildable on a machine without Python (e.g. a Windows build box).
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(STATUS "Python3 not found: script-driven tests will be registered but fail at run time")
  set(Python3_EXECUTABLE "python3")
endif()

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(ops/tests.cmake)
