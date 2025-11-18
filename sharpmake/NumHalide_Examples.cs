using Sharpmake;

[module: Sharpmake.Include("common.cs")]

namespace NumHalide
{
	[Sharpmake.Generate]
	public class NumHalideExamplesProject : CommonProject
	{
		public NumHalideExamplesProject()
		{
			Name = "NumHalide_Examples";
			SourceRootPath = RootPath + @"\examples";

			// Exclude legacy examples for now (contains duplicate main)
			SourceFilesExclude.Add(@"legacy_examples.cpp");
		}

		[Configure()]
		public void Configure(Configuration conf, NumHalideTarget target)
		{
			conf.SolutionFolder = "Examples";
			conf.Output = Configuration.OutputType.Exe;

			// NumHalide is header-only, just add include paths
			conf.IncludePaths.Add(@"[project.RootPath]\src");
			conf.IncludePaths.Add(@"[project.ExternPath]/stb");

			ConfigureHalide(conf, target);

			// Set working directory to project root for both debug and release
			conf.VcxprojUserFile = new Configuration.VcxprojUserFileSettings();
			conf.VcxprojUserFile.LocalDebuggerWorkingDirectory = @"[project.RootPath]\working_dir";

			// Link against Halide
			conf.LibraryPaths.Add(@"[project.ExternPath]/Halide/Libs/Release");
			conf.LibraryFiles.Add("Halide.lib");
		}
	}
}
