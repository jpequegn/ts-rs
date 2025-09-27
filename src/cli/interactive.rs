//! # Interactive REPL Mode
//!
//! This module provides an interactive Read-Eval-Print-Loop (REPL) interface
//! for exploring time series data.

use anyhow::{Result, Context};
use std::io::{self, Write};
use std::collections::VecDeque;
use colored::Colorize;

/// Interactive session state
pub struct InteractiveSession {
    /// Command history
    history: VecDeque<String>,
    /// Maximum history size
    max_history: usize,
    /// Current working data file
    current_file: Option<String>,
    /// Session context
    context: SessionContext,
}

/// Session context for maintaining state
#[derive(Default)]
pub struct SessionContext {
    /// Loaded data files
    pub files: Vec<String>,
    /// Current analysis results
    pub results: Vec<String>,
    /// Session variables
    pub variables: std::collections::HashMap<String, String>,
}

impl InteractiveSession {
    /// Create a new interactive session
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(100),
            max_history: 100,
            current_file: None,
            context: SessionContext::default(),
        }
    }

    /// Start the interactive REPL
    pub fn start(&mut self) -> Result<()> {
        self.print_welcome();

        loop {
            // Print prompt
            let prompt = if let Some(ref file) = self.current_file {
                format!("chronos [{}]> ", file.cyan())
            } else {
                "chronos> ".to_string()
            };

            print!("{}", prompt.green().bold());
            io::stdout().flush()?;

            // Read input
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            let input = input.trim();

            // Handle empty input
            if input.is_empty() {
                continue;
            }

            // Add to history
            self.add_to_history(input.to_string());

            // Process command
            match self.process_command(input) {
                Ok(should_exit) => {
                    if should_exit {
                        println!("{}", "Goodbye!".yellow());
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("{} {}", "Error:".red().bold(), e);
                }
            }
        }

        Ok(())
    }

    /// Process a command in the REPL
    fn process_command(&mut self, input: &str) -> Result<bool> {
        let parts: Vec<&str> = input.split_whitespace().collect();

        if parts.is_empty() {
            return Ok(false);
        }

        match parts[0] {
            "exit" | "quit" | "q" => return Ok(true),
            "help" | "?" => self.show_help(),
            "history" | "h" => self.show_history(),
            "clear" => self.clear_screen()?,
            "load" => self.load_file(&parts[1..])?,
            "files" => self.list_files(),
            "use" => self.use_file(&parts[1..])?,
            "info" => self.show_info()?,
            "set" => self.set_variable(&parts[1..])?,
            "get" => self.get_variable(&parts[1..])?,
            "run" => self.run_analysis(&parts[1..])?,
            _ => {
                // Try to run as a chronos command
                self.run_chronos_command(input)?;
            }
        }

        Ok(false)
    }

    /// Print welcome message
    fn print_welcome(&self) {
        println!("{}", "=".repeat(60).blue());
        println!("{}", "Welcome to Chronos Interactive Mode!".cyan().bold());
        println!("{}", "=".repeat(60).blue());
        println!();
        println!("Type {} for help, {} to exit", "help".yellow(), "quit".yellow());
        println!("Use {} to see available commands", "?".yellow());
        println!();
    }

    /// Show help for interactive mode
    fn show_help(&self) {
        println!("\n{}", "Interactive Mode Commands:".cyan().bold());
        println!("{}", "-".repeat(40));

        let commands = vec![
            ("help, ?", "Show this help message"),
            ("exit, quit, q", "Exit interactive mode"),
            ("history, h", "Show command history"),
            ("clear", "Clear the screen"),
            ("load <file>", "Load a data file"),
            ("files", "List loaded files"),
            ("use <file>", "Set current working file"),
            ("info", "Show information about current data"),
            ("set <var> <value>", "Set a session variable"),
            ("get <var>", "Get a session variable"),
            ("run <command>", "Run a chronos command"),
            ("", ""),
            ("Chronos Commands:", ""),
            ("stats [options]", "Run statistical analysis"),
            ("trend [options]", "Analyze trends"),
            ("seasonal [options]", "Detect seasonality"),
            ("anomaly [options]", "Detect anomalies"),
            ("forecast [options]", "Generate forecasts"),
            ("correlate [options]", "Correlation analysis"),
            ("plot [options]", "Create visualizations"),
            ("report [options]", "Generate reports"),
        ];

        for (cmd, desc) in commands {
            if cmd.is_empty() {
                println!();
            } else if desc.is_empty() {
                println!("\n{}", cmd.cyan().bold());
            } else {
                println!("  {:<20} {}", cmd.yellow(), desc);
            }
        }
        println!();
    }

    /// Add command to history
    fn add_to_history(&mut self, command: String) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(command);
    }

    /// Show command history
    fn show_history(&self) {
        println!("\n{}", "Command History:".cyan().bold());
        println!("{}", "-".repeat(40));

        for (i, cmd) in self.history.iter().enumerate() {
            println!("{:3}: {}", i + 1, cmd);
        }

        if self.history.is_empty() {
            println!("(No commands in history)");
        }
        println!();
    }

    /// Clear the screen
    fn clear_screen(&self) -> Result<()> {
        // Use ANSI escape codes to clear screen
        print!("\x1B[2J\x1B[1;1H");
        io::stdout().flush()?;
        Ok(())
    }

    /// Load a data file
    fn load_file(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Please specify a file to load"));
        }

        let filename = args[0];

        // Check if file exists
        if !std::path::Path::new(filename).exists() {
            return Err(anyhow::anyhow!("File not found: {}", filename));
        }

        self.context.files.push(filename.to_string());
        self.current_file = Some(filename.to_string());

        println!("{} Loaded file: {}", "✓".green().bold(), filename.cyan());
        Ok(())
    }

    /// List loaded files
    fn list_files(&self) {
        println!("\n{}", "Loaded Files:".cyan().bold());
        println!("{}", "-".repeat(40));

        if self.context.files.is_empty() {
            println!("(No files loaded)");
        } else {
            for (i, file) in self.context.files.iter().enumerate() {
                let marker = if Some(file.clone()) == self.current_file {
                    "*".green().bold()
                } else {
                    " ".normal()
                };
                println!("{} {}: {}", marker, i + 1, file);
            }
        }
        println!();
    }

    /// Set current working file
    fn use_file(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Please specify a file to use"));
        }

        let filename = args[0];

        if !self.context.files.contains(&filename.to_string()) {
            return Err(anyhow::anyhow!("File not loaded: {}. Use 'load' command first.", filename));
        }

        self.current_file = Some(filename.to_string());
        println!("{} Using file: {}", "✓".green().bold(), filename.cyan());
        Ok(())
    }

    /// Show information about current data
    fn show_info(&self) -> Result<()> {
        if let Some(ref file) = self.current_file {
            println!("\n{}", "Current File Information:".cyan().bold());
            println!("{}", "-".repeat(40));
            println!("File: {}", file.yellow());

            // TODO: Add actual file analysis here
            println!("(Analysis would be displayed here)");
        } else {
            println!("{}", "No file currently selected. Use 'load' or 'use' command.".yellow());
        }

        Ok(())
    }

    /// Set a session variable
    fn set_variable(&mut self, args: &[&str]) -> Result<()> {
        if args.len() < 2 {
            return Err(anyhow::anyhow!("Usage: set <variable> <value>"));
        }

        let var = args[0].to_string();
        let value = args[1..].join(" ");

        self.context.variables.insert(var.clone(), value.clone());
        println!("{} Set {} = {}", "✓".green().bold(), var.cyan(), value.yellow());

        Ok(())
    }

    /// Get a session variable
    fn get_variable(&self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            // Show all variables
            println!("\n{}", "Session Variables:".cyan().bold());
            println!("{}", "-".repeat(40));

            if self.context.variables.is_empty() {
                println!("(No variables set)");
            } else {
                for (key, value) in &self.context.variables {
                    println!("{} = {}", key.cyan(), value.yellow());
                }
            }
        } else {
            let var = args[0];
            if let Some(value) = self.context.variables.get(var) {
                println!("{} = {}", var.cyan(), value.yellow());
            } else {
                println!("{} Variable not found: {}", "!".red(), var);
            }
        }

        Ok(())
    }

    /// Run an analysis command
    fn run_analysis(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow::anyhow!("Please specify an analysis command"));
        }

        let command = args.join(" ");
        self.run_chronos_command(&command)?;

        Ok(())
    }

    /// Run a chronos command
    fn run_chronos_command(&mut self, command: &str) -> Result<()> {
        // Add current file if not specified
        let command = if let Some(ref file) = self.current_file {
            if !command.contains("--file") && !command.contains("-f") {
                format!("{} --file {}", command, file)
            } else {
                command.to_string()
            }
        } else {
            command.to_string()
        };

        println!("{} Running: chronos {}", "→".cyan().bold(), command);

        // TODO: Actually execute the command
        // For now, we'll just print what would be executed
        println!("(Command execution would happen here)");

        Ok(())
    }
}

/// Tab completion helper
pub struct TabCompleter {
    commands: Vec<String>,
    current_matches: Vec<String>,
}

impl TabCompleter {
    pub fn new() -> Self {
        let commands = vec![
            "help", "exit", "quit", "history", "clear",
            "load", "files", "use", "info", "set", "get", "run",
            "stats", "trend", "seasonal", "anomaly", "forecast",
            "correlate", "plot", "report",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            commands,
            current_matches: Vec::new(),
        }
    }

    /// Get completions for a partial command
    pub fn complete(&mut self, partial: &str) -> Vec<String> {
        self.current_matches = self.commands
            .iter()
            .filter(|cmd| cmd.starts_with(partial))
            .cloned()
            .collect();

        self.current_matches.clone()
    }
}

/// Command history manager
pub struct HistoryManager {
    history_file: std::path::PathBuf,
    max_entries: usize,
}

impl HistoryManager {
    pub fn new() -> Result<Self> {
        let history_file = dirs::home_dir()
            .context("Could not find home directory")?
            .join(".chronos_history");

        Ok(Self {
            history_file,
            max_entries: 1000,
        })
    }

    /// Load history from file
    pub fn load(&self) -> Result<VecDeque<String>> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let mut history = VecDeque::new();

        if self.history_file.exists() {
            let file = File::open(&self.history_file)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                if let Ok(cmd) = line {
                    history.push_back(cmd);
                    if history.len() > self.max_entries {
                        history.pop_front();
                    }
                }
            }
        }

        Ok(history)
    }

    /// Save history to file
    pub fn save(&self, history: &VecDeque<String>) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(&self.history_file)?;
        for cmd in history {
            writeln!(file, "{}", cmd)?;
        }

        Ok(())
    }
}