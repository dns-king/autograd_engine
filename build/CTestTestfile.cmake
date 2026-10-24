# CMake generated Testfile for 
# Source directory: C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine
# Build directory: C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test("autograd_tests" "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/Debug/autograd_tests.exe")
  set_tests_properties("autograd_tests" PROPERTIES  _CMAKE_TEST_BUILD_DEPENDS "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/Debug/autograd_tests.exe" _BACKTRACE_TRIPLES "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;22;add_test;C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test("autograd_tests" "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/Release/autograd_tests.exe")
  set_tests_properties("autograd_tests" PROPERTIES  _CMAKE_TEST_BUILD_DEPENDS "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/Release/autograd_tests.exe" _BACKTRACE_TRIPLES "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;22;add_test;C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test("autograd_tests" "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/MinSizeRel/autograd_tests.exe")
  set_tests_properties("autograd_tests" PROPERTIES  _CMAKE_TEST_BUILD_DEPENDS "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/MinSizeRel/autograd_tests.exe" _BACKTRACE_TRIPLES "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;22;add_test;C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test("autograd_tests" "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/RelWithDebInfo/autograd_tests.exe")
  set_tests_properties("autograd_tests" PROPERTIES  _CMAKE_TEST_BUILD_DEPENDS "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/build/RelWithDebInfo/autograd_tests.exe" _BACKTRACE_TRIPLES "C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;22;add_test;C:/Users/Manqoba/Downloads/PROJECTS/autograd_engine/autograd_engine/CMakeLists.txt;0;")
else()
  add_test("autograd_tests" NOT_AVAILABLE)
endif()
