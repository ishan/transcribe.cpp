# transcribe-sortformer-stream-parity: acceptance test for the Sortformer
# push-audio (STREAM-slot) path. Asserts that feeding a recording in arbitrary
# increments yields byte-identical speaker segments to one transcribe_run.

add_executable(transcribe-sortformer-stream-parity
    main.c
)

target_link_libraries(transcribe-sortformer-stream-parity
    PRIVATE
        transcribe
)

target_include_directories(transcribe-sortformer-stream-parity
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(main.c PROPERTIES COMPILE_OPTIONS
        "-Wno-sign-compare;-Wno-unused-function")
endif()
