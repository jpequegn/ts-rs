use clap::Parser;
use anyhow::Result;
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

// Use our CLI module
use chronos::cli::{Cli, Commands};
use chronos::cli::interactive::InteractiveSession;
use chronos::cli::commands::*;

fn setup_logging(verbose: bool, quiet: bool) {
    // Set up logging based on verbosity flags
    if verbose && !quiet {
        println!("{}", "Debug mode enabled".cyan().dimmed());
    }
}

fn setup_output_directory(output_dir: &Option<PathBuf>) -> Result<()> {
    if let Some(dir) = output_dir {
        if !dir.exists() {
            fs::create_dir_all(dir)?;
            println!("{}", format!("Created output directory: {}", dir.display()).green());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup global options
    setup_logging(cli.verbose, cli.quiet);
    setup_output_directory(&cli.output_dir)?;

    // Handle interactive mode
    if cli.interactive {
        if !cli.quiet {
            println!("{}", "🚀 Starting interactive mode...".cyan().bold());
        }
        let mut session = InteractiveSession::new();
        return session.start();
    }

    // Handle regular commands
    match cli.command {
        Some(Commands::Import(ref args)) => execute_import(args.clone(), &cli)?,
        Some(Commands::Stats(ref args)) => execute_stats(args.clone(), &cli)?,
        Some(Commands::Trend(ref args)) => execute_trend(args.clone(), &cli)?,
        Some(Commands::Seasonal(ref args)) => execute_seasonal(args.clone(), &cli)?,
        Some(Commands::Anomaly(ref args)) => execute_anomaly(args.clone(), &cli)?,
        Some(Commands::Forecast(ref args)) => execute_forecast(args.clone(), &cli)?,
        Some(Commands::Correlate(ref args)) => execute_correlate(args.clone(), &cli)?,
        Some(Commands::Plot(ref args)) => execute_plot(args.clone(), &cli)?,
        Some(Commands::Report(ref args)) => execute_report(args.clone(), &cli)?,
        None => {
            // No command provided, show help
            use clap::CommandFactory;
            let mut cmd = Cli::command();
            cmd.print_help()?;
            println!("\n\n{}", "💡 Use --help with any subcommand for detailed options".cyan());
            println!("{}", "💡 Use --interactive or -i to enter interactive mode".cyan());
        }
    }

    Ok(())
}
