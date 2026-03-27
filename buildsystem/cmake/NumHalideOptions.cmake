# NumHalideOptions.cmake
# Shared compiler settings for NumHalide targets

function(numhalide_set_target_options target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)

    target_compile_definitions(${target} PRIVATE NOMINMAX)

    if(MSVC)
        target_compile_options(${target} PRIVATE
            /bigobj
            /utf-8
            /GR-          # Disable RTTI (matches Sharpmake config)
            /EHsc         # Enable exceptions
            /FS           # Serialize PDB writes (required when /MP is active)
        )
        target_compile_definitions(${target} PRIVATE
            _ENABLE_EXTENDED_ALIGNED_STORAGE
        )

        # The pre-built Halide.dll (RelWithDebInfo) has a mixed CRT:
        #   Release C++ stdlib  (MSVCP140.dll)
        #   Debug C runtime     (VCRUNTIME140D.dll, ucrtbased.dll)
        # We must match both: use /MD for the C++ stdlib, and override
        # the default vcruntime/ucrt to link the debug variants.
        set_property(TARGET ${target} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
        )
        target_link_options(${target} PRIVATE
            /ignore:4098,4099,4217,4221
            /NODEFAULTLIB:vcruntime.lib
            /NODEFAULTLIB:ucrt.lib
        )
        target_link_libraries(${target} PRIVATE vcruntimed.lib ucrtd.lib)
    else()
        target_compile_options(${target} PRIVATE
            -fno-rtti
        )
    endif()
endfunction()

# Helper to copy Halide.dll next to a target after build
function(numhalide_copy_halide_dll target)
    if(WIN32 AND Halide_DLL)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${Halide_DLL}"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Copying Halide.dll for ${target}"
        )
    endif()
endfunction()
