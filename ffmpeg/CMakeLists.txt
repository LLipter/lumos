cmake_minimum_required(VERSION 3.12)
project(ffmpeg C)

set(CMAKE_C_STANDARD 99)

add_executable(ffmpeg main.c)

#link_directories(/usr/local/lib)
target_link_libraries(ffmpeg avcodec)
target_link_libraries(ffmpeg avdevice)
target_link_libraries(ffmpeg avfilter)
target_link_libraries(ffmpeg avformat)
target_link_libraries(ffmpeg avutil)
#target_link_libraries(ffmpeg postproc)
target_link_libraries(ffmpeg swresample)
target_link_libraries(ffmpeg swscale)

target_link_libraries(ffmpeg libz.dylib)
target_link_libraries(ffmpeg libbz2.1.0.dylib)
target_link_libraries(ffmpeg iconv)
target_link_libraries(ffmpeg lzma)

target_link_libraries(ffmpeg
        "-framework CoreFoundation"
        "-framework AudioToolbox"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework VideoToolbox"
        "-framework Security"
        )