using Sharpmake;

[module: Sharpmake.Include("common.cs")]

namespace NumHalide
{
	// Base class for example projects
	public class ExampleProjectBase : CommonProject
	{
		public ExampleProjectBase()
		{
			// Include stbi_impl from _common
			AdditionalSourceRootPaths.Add(RootPath + @"\examples\_common");
		}

		[Configure()]
		public void ConfigureExample(Configuration conf, NumHalideTarget target)
		{
			conf.SolutionFolder = "Examples";
			conf.Output = Configuration.OutputType.Exe;

			// NumHalide is header-only, just add include paths
			conf.IncludePaths.Add(@"[project.RootPath]\src");
			conf.IncludePaths.Add(@"[project.RootPath]\examples\_common");
			conf.IncludePaths.Add(@"[project.ExternPath]/stb");

			ConfigureHalide(conf, target);

			// Set working directory to working_dir (use VS macro for consistency)
			conf.VcxprojUserFile = new Configuration.VcxprojUserFileSettings();
			conf.VcxprojUserFile.LocalDebuggerWorkingDirectory = @"$(SolutionDir)working_dir";

			// Link against Halide
			conf.LibraryPaths.Add(@"[project.ExternPath]/Halide/Libs/Release");
			conf.LibraryFiles.Add("Halide.lib");
		}
	}

	[Sharpmake.Generate]
	public class Example00_Gradient : ExampleProjectBase
	{
		public Example00_Gradient()
		{
			Name = "Example_00_gradient";
			SourceRootPath = RootPath + @"\examples\00_gradient";
		}
	}

	[Sharpmake.Generate]
	public class Example01_ShapeDebug : ExampleProjectBase
	{
		public Example01_ShapeDebug()
		{
			Name = "Example_01_shape_debug";
			SourceRootPath = RootPath + @"\examples\01_shape_debug";
		}
	}

	[Sharpmake.Generate]
	public class Example02_Factories : ExampleProjectBase
	{
		public Example02_Factories()
		{
			Name = "Example_02_factories";
			SourceRootPath = RootPath + @"\examples\02_factories";
		}
	}

	[Sharpmake.Generate]
	public class Example03_Stacking : ExampleProjectBase
	{
		public Example03_Stacking()
		{
			Name = "Example_03_stacking";
			SourceRootPath = RootPath + @"\examples\03_stacking";
		}
	}

	[Sharpmake.Generate]
	public class Example04_Broadcasting : ExampleProjectBase
	{
		public Example04_Broadcasting()
		{
			Name = "Example_04_broadcasting";
			SourceRootPath = RootPath + @"\examples\04_broadcasting";
		}
	}

	[Sharpmake.Generate]
	public class Example05_Reductions : ExampleProjectBase
	{
		public Example05_Reductions()
		{
			Name = "Example_05_reductions";
			SourceRootPath = RootPath + @"\examples\05_reductions";
		}
	}

	[Sharpmake.Generate]
	public class Example06_Slicing : ExampleProjectBase
	{
		public Example06_Slicing()
		{
			Name = "Example_06_slicing";
			SourceRootPath = RootPath + @"\examples\06_slicing";
		}
	}

	[Sharpmake.Generate]
	public class Example07_Random : ExampleProjectBase
	{
		public Example07_Random()
		{
			Name = "Example_07_random";
			SourceRootPath = RootPath + @"\examples\07_random";
		}
	}

	[Sharpmake.Generate]
	public class Example08_Matmul : ExampleProjectBase
	{
		public Example08_Matmul()
		{
			Name = "Example_08_matmul";
			SourceRootPath = RootPath + @"\examples\08_matmul";
		}
	}

	[Sharpmake.Generate]
	public class Example09_Masks : ExampleProjectBase
	{
		public Example09_Masks()
		{
			Name = "Example_09_masks";
			SourceRootPath = RootPath + @"\examples\09_masks";
		}
	}

	[Sharpmake.Generate]
	public class Example10_Scheduling : ExampleProjectBase
	{
		public Example10_Scheduling()
		{
			Name = "Example_10_scheduling";
			SourceRootPath = RootPath + @"\examples\10_scheduling";
		}
	}

	[Sharpmake.Generate]
	public class Example11_Statistics : ExampleProjectBase
	{
		public Example11_Statistics()
		{
			Name = "Example_11_statistics";
			SourceRootPath = RootPath + @"\examples\11_statistics";
		}
	}

	[Sharpmake.Generate]
	public class Example12_BoolReduce : ExampleProjectBase
	{
		public Example12_BoolReduce()
		{
			Name = "Example_12_bool_reduce";
			SourceRootPath = RootPath + @"\examples\12_bool_reduce";
		}
	}

	[Sharpmake.Generate]
	public class Example13_ManipulationExt : ExampleProjectBase
	{
		public Example13_ManipulationExt()
		{
			Name = "Example_13_manipulation_ext";
			SourceRootPath = RootPath + @"\examples\13_manipulation_ext";
		}
	}

	[Sharpmake.Generate]
	public class Example14_Comparisons : ExampleProjectBase
	{
		public Example14_Comparisons()
		{
			Name = "Example_14_comparisons";
			SourceRootPath = RootPath + @"\examples\14_comparisons";
		}
	}

	[Sharpmake.Generate]
	public class Example15_SetOps : ExampleProjectBase
	{
		public Example15_SetOps()
		{
			Name = "Example_15_set_ops";
			SourceRootPath = RootPath + @"\examples\15_set_ops";
		}
	}

	[Sharpmake.Generate]
	public class Example16_Sorting : ExampleProjectBase
	{
		public Example16_Sorting()
		{
			Name = "Example_16_sorting";
			SourceRootPath = RootPath + @"\examples\16_sorting";
		}
	}

	[Sharpmake.Generate]
	public class Example17_LinalgExt : ExampleProjectBase
	{
		public Example17_LinalgExt()
		{
			Name = "Example_17_linalg_ext";
			SourceRootPath = RootPath + @"\examples\17_linalg_ext";
		}
	}

	[Sharpmake.Generate]
	public class Example18_FFT : ExampleProjectBase
	{
		public Example18_FFT()
		{
			Name = "Example_18_fft";
			SourceRootPath = RootPath + @"\examples\18_fft";
		}
	}

	[Sharpmake.Generate]
	public class Example19_Convolution : ExampleProjectBase
	{
		public Example19_Convolution()
		{
			Name = "Example_19_convolution";
			SourceRootPath = RootPath + @"\examples\19_convolution";
		}
	}

	[Sharpmake.Generate]
	public class Example20_Interpolation : ExampleProjectBase
	{
		public Example20_Interpolation()
		{
			Name = "Example_20_interpolation";
			SourceRootPath = RootPath + @"\examples\20_interpolation";
		}
	}
}
