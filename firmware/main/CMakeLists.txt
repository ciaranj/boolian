idf_component_register(SRCS "main.c" INCLUDE_DIRS ".")

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/secrets.h")
    message(FATAL_ERROR "firmware/main/secrets.h is missing: copy secrets.example.h to secrets.h and fill it in "
                        "(it's gitignored, so your Wi-Fi password stays out of the repo)")
endif()
