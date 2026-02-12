using Sharpmake;

[module: Sharpmake.Include("common.cs")]

namespace NumHalide
{
	[Sharpmake.Generate]
	public class NumHalideProject : CommonProject
	{
		public NumHalideProject()
		{
			Name = "NumHalide";
			SourceRootPath = RootPath + @"\src";
		}

		[Configure()]
		public void Configure(Configuration conf, NumHalideTarget target)
		{
			conf.SolutionFolder = "Core";

			// Static library
			conf.Output = Configuration.OutputType.Lib;

			ConfigureHalide(conf, target);

			conf.IncludePaths.Add(@"[project.RootPath]\src");
		}
	}
}
