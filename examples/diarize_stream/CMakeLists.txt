# transcribe-diarize-stream: streaming diarizer with dynamic windowing for the Sortformer
# push-audio (STREAM-slot) path.
# increments yields byte-identical speaker segments to one transcribe_run.

add_executable(transcribe-diarize-stream
    main.c
)

target_link_libraries(transcribe-diarize-stream
    PRIVATE
        transcribe
)

target_include_directories(transcribe-diarize-stream
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(main.c PROPERTIES COMPILE_OPTIONS
        "-Wno-sign-compare;-Wno-unused-function")
endif()
