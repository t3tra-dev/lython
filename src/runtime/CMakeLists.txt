add_library(lython_runtime STATIC dummy.c)

target_include_directories(lython_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
