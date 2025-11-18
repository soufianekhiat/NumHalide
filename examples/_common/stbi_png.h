/// @file stbi_png.h
/// @brief Simple PNG output wrapper using stb_image_write
///
/// Provides a convenient save_png() function for writing Halide buffers to PNG files.
/// Note: The implementation is in stbi_impl.cpp

#pragma once

#include <HalideBuffer.h>
#include "stb_image_write.h"

/// @brief Save a Halide buffer to PNG file
/// @param buf Buffer containing image data (width x height x channels)
/// @param path Output file path
/// @return true if write succeeded, false otherwise
inline bool save_png(const Halide::Runtime::Buffer<uint8_t>& buf, const char* path) {
	int w = buf.width();
	int h = buf.height();
	int c = buf.channels();
	return stbi_write_png(path, w, h, c, buf.data(), w * c) != 0;
}
