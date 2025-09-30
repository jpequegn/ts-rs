use clap::Parser;
use anyhow::Result;
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

// Use our CLI module
use chronos::cli::{Cli, Commands};
use chronos::cli::interactive::InteractiveSession;
use chronos::cli::commands::*;
use chronos::config::{Config, ConfigLoader};

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

fn load_configuration(config_path: &Option<PathBuf>, verbose: bool) -> Result<Config> {
    let loader = ConfigLoader::new();

    let config = match config_path {
        Some(path) => {
            if verbose {
                println!("{}", format!("Loading configuration from: {}", path.display()).cyan().dimmed());
            }
            loader.load_with_file(path)?
        }
        None => {
            if verbose {
                println!("{}", "Loading default configuration...".cyan().dimmed());
            }
            loader.load()?
        }
    };

    if verbose {
        println!("{}", format!("Configuration loaded successfully. Active profile: {}", config.metadata.active_profile).cyan().dimmed());
    }

    Ok(config)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup global options
    setup_logging(cli.verbose, cli.quiet);
    setup_output_directory(&cli.output_dir)?;

    // Load configuration (unless it's a config command that might create the config file)
    let _config = match &cli.command {
        Some(Commands::Config(_)) => {
            // For config commands, we might be creating the config file, so loading might fail
            // This is handled within the config command itself
            None
        }
        _ => {
            // For other commands, try to load configuration
            match load_configuration(&cli.config, cli.verbose && !cli.quiet) {
                Ok(config) => Some(config),
                Err(e) => {
                    if cli.verbose && !cli.quiet {
                        println!("{}", format!("Warning: Could not load configuration: {}. Using defaults.", e).yellow());
                    }
                    None
                }
            }
        }
    };

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
        Some(Commands::Config(ref args)) => execute_config(args.clone(), &cli)?,
        Some(Commands::Plugin(ref args)) => {
            println!("{}", "🔌 Plugin management feature coming soon...".cyan());
            println!("Plugin command received: {:?}", args);
        },
        Some(Commands::Quality(ref args)) => execute_quality(args.clone(), &cli)?,
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
