using Sharpmake;

[module: Sharpmake.Include("common.cs")]

namespace NumHalide
{
	[Sharpmake.Generate]
	public class NumHalideTestsProject : CommonProject
	{
		public NumHalideTestsProject()
		{
			Name = "NumHalide_Tests";
			SourceRootPath = RootPath + @"\tests";
		}

		[Configure()]
		public void Configure(Configuration conf, NumHalideTarget target)
		{
			conf.SolutionFolder = "Tests";
			conf.Output = Configuration.OutputType.Exe;

			// NumHalide is header-only, no library dependency needed
			// Just add include paths
			conf.IncludePaths.Add(@"[project.RootPath]\src");

			ConfigureHalide(conf, target);

			// Set working directory to project root for both debug and release
			conf.VcxprojUserFile = new Configuration.VcxprojUserFileSettings();
			conf.VcxprojUserFile.LocalDebuggerWorkingDirectory = @"[project.RootPath]\working_dir";

			// Use static GoogleTest to avoid DLL issues
			conf.Defines.Add("GTEST_LINKED_AS_SHARED_LIBRARY=0");

			// vcpkg will auto-link gtest based on vcpkg.json manifest at project root
			// Make sure vcpkg.json exists in project root directory
		}
	}
}
