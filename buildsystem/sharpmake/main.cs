using Sharpmake;

[module: Sharpmake.Include("common.cs")]
[module: Sharpmake.Include("NumHalide.cs")]
[module: Sharpmake.Include("NumHalide_Examples.cs")]
[module: Sharpmake.Include("NumHalide_Tests.cs")]

namespace NumHalide
{
	[Sharpmake.Generate]
	public class NumHalideSolution : CommonSolution
	{
		public NumHalideSolution()
		{
			Name = "NumHalide";
			AddTargets(new NumHalideTarget(
				Platform.win64,
				Optimization.Debug | Optimization.Release));
		}

		[Configure()]
		public void ConfigureAll(Configuration conf, NumHalideTarget target)
		{
			conf.Name = "[target.Optimization]";
			conf.SolutionFileName = "[solution.Name]_[target.Platform]";
			conf.SolutionPath = SolutionRootPath;

			conf.AddProject<NumHalideProject>(target);
			conf.AddProject<NumHalideTestsProject>(target);

			// Example projects
			conf.AddProject<Example00_Gradient>(target);
			conf.AddProject<Example01_ShapeDebug>(target);
			conf.AddProject<Example02_Factories>(target);
			conf.AddProject<Example03_Stacking>(target);
			conf.AddProject<Example04_Broadcasting>(target);
			conf.AddProject<Example05_Reductions>(target);
			conf.AddProject<Example06_Slicing>(target);
			conf.AddProject<Example07_Random>(target);
			conf.AddProject<Example08_Matmul>(target);
			conf.AddProject<Example09_Masks>(target);
			conf.AddProject<Example10_Scheduling>(target);
			conf.AddProject<Example11_Statistics>(target);
			conf.AddProject<Example12_BoolReduce>(target);
			conf.AddProject<Example13_ManipulationExt>(target);
			conf.AddProject<Example14_Comparisons>(target);
			conf.AddProject<Example15_SetOps>(target);
			conf.AddProject<Example16_Sorting>(target);
			conf.AddProject<Example18_FFT>(target);
			conf.AddProject<Example17_LinalgExt>(target);
			conf.AddProject<Example19_Convolution>(target);
			conf.AddProject<Example20_Interpolation>(target);
			conf.AddProject<Example21_Trigonometry>(target);
			conf.AddProject<Example22_Math>(target);
			conf.AddProject<Example23_Cumulative>(target);
			conf.AddProject<Example24_Splitting>(target);
			conf.AddProject<Example25_Closeness>(target);
			conf.AddProject<Example26_StatisticsExt>(target);
			conf.AddProject<Example27_RandomExt>(target);
			conf.AddProject<Example28_ArrayCompare>(target);
			conf.AddProject<Example29_Bitwise>(target);
			conf.AddProject<Example30_Windows>(target);
			conf.AddProject<Example31_RFFT>(target);
			conf.AddProject<Example32_Gradient>(target);
			conf.AddProject<Example33_Morphology>(target);
			conf.AddProject<Example34_Color>(target);
			conf.AddProject<Example35_Polynomial>(target);
			conf.AddProject<Example36_Distance>(target);
			conf.AddProject<Example37_Stencil>(target);
			conf.AddProject<Example38_Histogram>(target);
			conf.AddProject<Example39_Spectral>(target);
			conf.AddProject<Example40_Threshold>(target);
		}
	}

	public class Main
	{
		[Sharpmake.Main]
		public static void SharpmakeMain(Sharpmake.Arguments arguments)
		{
			Sharpmake.KitsRootPaths.SetUseKitsRootForDevEnv(
				DevEnv.vs2022,
				KitsRootEnum.KitsRoot10,
				Sharpmake.Options.Vc.General.WindowsTargetPlatformVersion.Latest);

			arguments.Generate<NumHalideSolution>();
		}
	}
}
