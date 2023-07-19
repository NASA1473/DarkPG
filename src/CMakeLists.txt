add_library( lime_feature SHARED lime.cpp)
add_library( yolo_feature SHARED yolo.cpp)
add_library( inference_feature SHARED inference.cpp)

include_directories(/home/kylin/fftw/include/)
link_directories(/home/kylin/fftw/lib/)

target_link_libraries( lime_feature
    ${THIRD_PARTY_LIBS}
    fftw3f m
)

target_link_libraries( yolo_feature
    ${THIRD_PARTY_LIBS}
    fftw3f m
)

target_link_libraries( inference_feature
    ${THIRD_PARTY_LIBS}
    fftw3f m
)   