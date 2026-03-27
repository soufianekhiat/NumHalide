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

			conf.IncludePaths.Add(@"[project.RootPath]\src");

			ConfigureHalide(conf, target);

			// GoogleTest — always use the Release flavor of gtest.dll.
			// Our exe links /MD even in Debug (forced by Halide.dll's mixed CRT), so we
			// must never use the vcpkg debug gtest (/MDd) — that would cause an
			// _ITERATOR_DEBUG_LEVEL mismatch and a crash in std::string across the boundary.
			string vcpkgRoot = @"[project.ProjectRootPath]\build\vcpkg_installed\x64-windows";
			conf.IncludePaths.Add(vcpkgRoot + @"\include");
			conf.LibraryPaths.Add(vcpkgRoot + @"\lib");
			conf.LibraryPaths.Add(vcpkgRoot + @"\lib\manual-link");
			conf.LibraryFiles.Add("gtest.lib");
			conf.LibraryFiles.Add("gtest_main.lib");
			conf.TargetCopyFiles.Add(vcpkgRoot + @"\bin\gtest.dll");
			conf.TargetCopyFiles.Add(vcpkgRoot + @"\bin\gtest_main.dll");

			// GTest DLL flavour — vcpkg builds GTest as a shared library
			conf.Defines.Add("GTEST_LINKED_AS_SHARED_LIBRARY=1");

			// Working directory for the debugger
			conf.VcxprojUserFile = new Configuration.VcxprojUserFileSettings();
			conf.VcxprojUserFile.LocalDebuggerWorkingDirectory = @"[project.RootPath]\working_dir";
		}
	}
}
