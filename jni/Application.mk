APP_ABI           := arm64-v8a
LIBCXX_USE_GABIXX := true
APP_STL           := c++_shared
APP_PLATFORM      := android-29
APP_SUPPORT_FLEXIBLE_PAGE_SIZES := true
APP_CFLAGS        += -march=armv8.2-a+fp16+dotprod+i8mm
APP_CPPFLAGS      += -march=armv8.2-a+fp16+dotprod+i8mm
