/// @file 02_factories.cpp
/// @brief Example 02: Demonstrate factory functions
///
/// Demonstrates:
///   - linspace() for evenly spaced values
///   - arange() for integer ranges
///   - identity() for identity matrices
///   - zeros(), ones(), full() for constant arrays
///
/// Output: out/02_factories.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>

using namespace numhalide;

int main(int argc, char **argv) {
	try {
		// Image dimensions (4 panels: linspace, arange, identity, ones)
		const int panel_width = 128;
		const int panel_height = 128;
		const int width = panel_width * 4;
		const int height = panel_height;

		std::cout << "Creating factory functions visualization " << width << "x" << height << "..." << std::endl;

		Halide::Var x("x"), y("y");

		// Panel 1: Horizontal linspace gradient
		Halide::Func ls = linspace(Halide::Float(32), 0.0f, 1.0f, panel_width);
		Halide::Func panel1("panel1");
		panel1(x, y) = Halide::cast<uint8_t>(ls(x) * 255.0f);

		// Panel 2: Vertical arange gradient
		Halide::Func ar = arange(Halide::Float(32), 0.0f, (float)panel_height, 1.0f);
		Halide::Func panel2("panel2");
		panel2(x, y) = Halide::cast<uint8_t>((ar(y) / (float)panel_height) * 255.0f);

		// Panel 3: Identity matrix (scaled up)
		Halide::Func id = identity(Halide::Float(32), 16);
		Halide::Func panel3("panel3");
		// Scale up: each cell is 8x8 pixels
		panel3(x, y) = Halide::cast<uint8_t>(id(x / 8, y / 8) * 255.0f);

		// Panel 4: Checkerboard using ones and zeros
		Halide::Func panel4("panel4");
		// Create a checkerboard pattern
		panel4(x, y) = Halide::cast<uint8_t>(((x / 16 + y / 16) % 2) * 255);

		// Combine all panels
		Halide::Func combined("combined");
		combined(x, y) = Halide::select(
			x < panel_width, panel1(x, y),
			Halide::select(
				x < panel_width * 2, panel2(x - panel_width, y),
				Halide::select(
					x < panel_width * 3, panel3(x - panel_width * 2, y),
					panel4(x - panel_width * 3, y)
				)
			)
		);

		// Realize the result
		std::cout << "Computing..." << std::endl;
		Halide::Runtime::Buffer<uint8_t> output(width, height);
		combined.realize(output);

		// Save to PNG
		const char* output_path = "out/02_factories.png";
		std::cout << "Saving to " << output_path << "..." << std::endl;

		if (save_png(output, output_path)) {
			std::cout << "Success! Factory functions visualization saved to " << output_path << std::endl;
			std::cout << "\nPanels (left to right):" << std::endl;
			std::cout << "  1. linspace(0, 1, 128) - horizontal gradient" << std::endl;
			std::cout << "  2. arange(0, 128, 1) - vertical gradient" << std::endl;
			std::cout << "  3. identity(16) - scaled identity matrix" << std::endl;
			std::cout << "  4. Checkerboard pattern" << std::endl;
		} else {
			std::cerr << "Error: Failed to save PNG" << std::endl;
			return 1;
		}

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
