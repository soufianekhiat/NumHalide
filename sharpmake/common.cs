using System;
using Sharpmake;
using System.IO;
using System.Collections.Generic;

namespace NumHalide
{
	static class Extern
	{
		static public string RootPath = @"[project.SharpmakeCsPath]\..";
		static public string ExternPath = RootPath + @"\extern";
	}

	public class NumHalideTarget : Target
	{
		public NumHalideTarget()
			: base()
		{ }

		public NumHalideTarget(Platform platform, Optimization optimization)
			: base(platform, DevEnv.vs2022, optimization)
		{
		}
	}

	public class CommonSolution : Sharpmake.Solution
	{
		public string SolutionRootPath = @"[solution.SharpmakeCsPath]\..";

		public CommonSolution()
			: base(typeof(NumHalideTarget))
		{ }
	}

	public class CommonProject : Sharpmake.Project
	{
		public string ProjectRootPath = @"[project.SharpmakeCsPath]\..";
		public string TmpPath = @"[project.ProjectRootPath]\tmp";
		public string ExternPath = @"[project.ProjectRootPath]\extern";
		public string ProjectsPath = @"[project.ProjectRootPath]\projects";
		public string OutputPath = @"[project.ProjectRootPath]\working_dir";

		public CommonProject()
			: base(typeof(NumHalideTarget))
		{
			RootPath = ProjectRootPath;
			SourceFilesExtensions.Add(".inl", ".hpp");

			// Enable vcpkg
			IsFileNameToLower = false;

			AddTargets(new NumHalideTarget(
				Platform.win64,
				Optimization.Debug | Optimization.Release));
		}

		public void ConfigureHalide(Configuration conf, NumHalideTarget target)
		{
			conf.IncludePaths.Add(@"[project.ExternPath]/Halide/include");
			conf.LibraryPaths.Add(@"[project.ExternPath]/Halide/Libs/Release");
			conf.LibraryFiles.Add("Halide.lib");
		}

		[Configure()]
		public virtual void ConfigureAll(Configuration conf, NumHalideTarget target)
		{
			conf.Name = "[target.Optimization]";
			conf.ProjectFileName = "[project.Name]_[target.DevEnv]_[target.Platform]";
			conf.ProjectPath = ProjectsPath + @"\[target.Platform]\[project.Name]";
			conf.IntermediatePath = @"[project.TmpPath]\[target.Platform]\[project.Name]\[target.Optimization]";
			conf.TargetPath = @"[project.OutputPath]\[target.Optimization]";
			conf.TargetLibraryPath = @"[project.OutputPath]\[target.Optimization]";

			conf.IncludePaths.Add(@"[project.ProjectRootPath]");
			conf.IncludePaths.Add(@"[project.ProjectRootPath]\src");
			conf.IncludePaths.Add(@"[project.ExternPath]");
			conf.IncludePaths.Add(@"[project.ExternPath]/stb");

			conf.Output = Configuration.OutputType.Lib;

			conf.AdditionalCompilerOptions.Add("/bigobj");
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.Inline.AnySuitable);
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.RTTI.Disable);
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.Exceptions.Enable);
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.CppLanguageStandard.CPP20);
			conf.Options.Add(Sharpmake.Options.Vc.Linker.GenerateDebugInformation.Enable);

			conf.AdditionalCompilerOptions.Add("/utf-8");
			conf.Defines.Add("NOMINMAX");
		}

		[Configure(Platform.win64)]
		public virtual void ConfigureWindows(Configuration conf, NumHalideTarget target)
		{
			conf.Defines.Add("_ENABLE_EXTENDED_ALIGNED_STORAGE");
			conf.AdditionalLinkerOptions.Add("/ignore:4098,4099,4217,4221");
		}

		[Configure(Optimization.Debug)]
		public virtual void ConfigureDebug(Configuration conf, NumHalideTarget target)
		{
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.Inline.Disable);
			// Use Release runtime even in Debug to match pre-built Halide.dll
			// Halide binaries in extern/Halide/Libs/Release were built with Release CRT
			// Mixing Debug CRT (MSVCRTD) with Release CRT (MSVCRT) causes heap corruption
			// when STL objects cross the DLL boundary
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.RuntimeLibrary.MultiThreadedDLL);
		}

		[Configure(Optimization.Release)]
		public virtual void ConfigureRelease(Configuration conf, NumHalideTarget target)
		{
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.Optimization.MaximizeSpeed);
			conf.Options.Add(Sharpmake.Options.Vc.General.WholeProgramOptimization.LinkTime);
			conf.Options.Add(Sharpmake.Options.Vc.Linker.LinkTimeCodeGeneration.UseLinkTimeCodeGeneration);
			conf.Options.Add(Sharpmake.Options.Vc.Compiler.RuntimeLibrary.MultiThreadedDLL);
		}

		[Configure(OutputType.Lib)]
		public virtual void OutputTypeLib(Configuration conf, NumHalideTarget target)
		{
			if (target.Platform == Platform.win64)
			{
				conf.Options.Add(Options.Vc.General.PreferredToolArchitecture.x64);
			}
		}
	}
}
