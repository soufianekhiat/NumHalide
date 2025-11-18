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
			conf.AddProject<NumHalideExamplesProject>(target);
			conf.AddProject<NumHalideTestsProject>(target);
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
