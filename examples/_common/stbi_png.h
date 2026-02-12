/// @file stbi_png.h
/// @brief Simple PNG output wrapper using stb_image_write
///
/// Provides a convenient save_png() function for writing Halide buffers to PNG files.
/// Note: The implementation is in stbi_impl.cpp

#pragma once

#include <HalideBuffer.h>
#include <vector>
#include "stb_image_write.h"

/// @brief Save a Halide buffer to PNG file
/// @param buf Buffer containing image data (width x height) or (width x height x channels)
/// @param path Output file path
/// @return true if write succeeded, false otherwise
/// @note Halide buffers are planar (RRR...GGG...BBB), stbi expects interleaved (RGBRGB...)
inline bool save_png(const Halide::Runtime::Buffer<uint8_t>& buf, const char* path) {
	int w = buf.width();
	int h = buf.height();
	int c = buf.channels();

	// For single-channel images, data is already in correct format
	if (c == 1) {
		return stbi_write_png(path, w, h, c, buf.data(), w) != 0;
	}

	// For multi-channel images, convert from planar to interleaved
	std::vector<uint8_t> interleaved(w * h * c);
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			for (int ch = 0; ch < c; ++ch) {
				interleaved[(y * w + x) * c + ch] = buf(x, y, ch);
			}
		}
	}

	return stbi_write_png(path, w, h, c, interleaved.data(), w * c) != 0;
}
