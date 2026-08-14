# kernscope benchmark/config database (JSONL) - plan todo 8.
#
# libs/db: append-only JSONL benchmark/config store keyed by
# (gpu_arch, kernel_family, shape_hash, config) with a best-by-median query,
# a forward-compatible versioned schema and corrupt-line resilience (design
# per Triton autotune disk cache + TVM MetaSchedule JSONDatabase,
# docs/research/autotuner-research-brief.md §1-§2).
#
# Zero external dependencies by design: nlohmann/json is not in the devenv
# profile and todo 8 must not modify devenv files, so the JSON value model is
# self-contained in kernscope/db.h (rationale documented there).
add_library(kernscope_db STATIC src/db.cpp)
target_include_directories(kernscope_db PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_db PUBLIC cxx_std_20)
add_library(kernscope::db ALIAS kernscope_db)

# gtest unit tests, registered with the ctest hardware-gating convention
# (LABELS no-gpu: pure CPU, no GPU required - root CMakeLists.txt enables
# testing unconditionally, so tests register unconditionally too).
find_package(GTest REQUIRED)
add_executable(kernscope-db-test tests/db_test.cpp)
target_link_libraries(kernscope-db-test PRIVATE kernscope::db GTest::gtest_main)
add_test(NAME db COMMAND kernscope-db-test)
set_tests_properties(db PROPERTIES LABELS "no-gpu")
set_target_properties(kernscope-db-test PROPERTIES FOLDER "Tests")
