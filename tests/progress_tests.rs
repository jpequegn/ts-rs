//! Unit tests for progress tracking module

use chronos::performance::progress::*;
use chronos::config::PerformanceConfig;
use std::time::Duration;
use std::thread;

fn create_test_config() -> PerformanceConfig {
    PerformanceConfig {
        enable_optimization: true,
        max_memory_mb: 100,
        chunk_size: 1000,
        num_threads: Some(2),
        max_cache_size_mb: 50,
        progress_threshold: 10, // Low threshold for testing
        enable_database: true,
        cache_directory: Some(std::env::temp_dir().join("chronos_test_cache")),
    }
}

#[test]
fn test_progress_tracker_creation() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let operations = tracker.get_active_operations();
    assert_eq!(operations.len(), 0);
}

#[test]
fn test_progress_bar_creation_with_threshold() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    // Below threshold - should create silent progress bar
    let pb_silent = tracker.create_progress_bar("test_small", 5);
    assert!(!pb_silent.is_cancellable());

    // Above threshold - should create active progress bar
    let pb_active = tracker.create_progress_bar("test_large", 50);
    assert!(pb_active.is_cancellable());
}

#[test]
fn test_progress_bar_operations() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let pb = tracker.create_progress_bar("test_operation", 100);

    // Test initial state
    let progress = pb.get_progress();
    assert_eq!(progress.current, 0);
    assert_eq!(progress.total, 100);
    assert_eq!(progress.percentage, 0.0);

    // Test increment
    pb.inc();
    let progress = pb.get_progress();
    assert_eq!(progress.current, 1);
    assert_eq!(progress.percentage, 1.0);

    // Test add
    pb.add(9);
    let progress = pb.get_progress();
    assert_eq!(progress.current, 10);
    assert_eq!(progress.percentage, 10.0);

    // Test set position
    pb.set_position(50);
    let progress = pb.get_progress();
    assert_eq!(progress.current, 50);
    assert_eq!(progress.percentage, 50.0);

    // Test finish
    pb.finish();
    assert!(pb.is_finished());
}

#[test]
fn test_progress_bar_silent() {
    let pb = ProgressBar::new_silent("test_silent", 100);

    assert!(!pb.is_cancellable());
    assert!(!pb.is_cancelled());

    pb.inc();
    let progress = pb.get_progress();
    assert_eq!(progress.current, 1);
    assert_eq!(progress.percentage, 1.0);

    pb.finish();
    assert!(pb.is_finished());
}

#[test]
fn test_progress_bar_cancellation() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let pb = tracker.create_progress_bar("cancellable_test", 100);

    assert!(pb.is_cancellable());
    assert!(!pb.is_cancelled());

    pb.cancel();
    assert!(pb.is_cancelled());
}

#[test]
fn test_progress_spinner() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let spinner = tracker.create_spinner("test_spinner");

    spinner.set_message("Processing...");

    let elapsed = spinner.elapsed();
    assert!(elapsed.as_millis() >= 0);

    spinner.finish();
}

#[test]
fn test_progress_spinner_silent() {
    let spinner = ProgressSpinner::new_silent("silent_spinner");

    spinner.set_message("Silent processing...");

    let elapsed = spinner.elapsed();
    assert!(elapsed.as_millis() >= 0);

    spinner.finish_with_message("Silent completed");
}

#[test]
fn test_progress_tracker_active_operations() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    // Create multiple progress bars
    let pb1 = tracker.create_progress_bar("operation1", 100);
    let pb2 = tracker.create_progress_bar("operation2", 200);

    let operations = tracker.get_active_operations();
    assert_eq!(operations.len(), 2);

    // Verify operation details
    let op1 = operations.iter().find(|op| op.name == "operation1").unwrap();
    assert_eq!(op1.progress_percentage, 0.0);
    assert!(op1.is_cancellable);
    assert!(!op1.is_cancelled);

    // Update progress and check again
    pb1.set_position(50);
    let operations = tracker.get_active_operations();
    let op1 = operations.iter().find(|op| op.name == "operation1").unwrap();
    assert_eq!(op1.progress_percentage, 50.0);
}

#[test]
fn test_progress_tracker_cancellation() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let _pb = tracker.create_progress_bar("cancellable_operation", 100);

    // Cancel by name
    assert!(tracker.cancel_operation("cancellable_operation"));
    assert!(!tracker.cancel_operation("nonexistent_operation"));

    let operations = tracker.get_active_operations();
    let op = operations.iter().find(|op| op.name == "cancellable_operation").unwrap();
    assert!(op.is_cancelled);
}

#[test]
fn test_progress_tracker_cancel_all() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let _pb1 = tracker.create_progress_bar("operation1", 100);
    let _pb2 = tracker.create_progress_bar("operation2", 200);

    tracker.cancel_all();

    let operations = tracker.get_active_operations();
    for op in operations {
        assert!(op.is_cancelled);
    }
}

#[test]
fn test_progress_aware_execution_success() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let result = tracker.execute_with_progress("test_task", 10, |pb| {
        for i in 0..10 {
            pb.inc();
            thread::sleep(Duration::from_millis(1));
        }
        Ok(42)
    });

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_progress_aware_execution_failure() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let result = tracker.execute_with_progress("failing_task", 10, |_pb| {
        Err("Task failed".into())
    });

    assert!(result.is_err());
}

#[test]
fn test_progress_aware_spinner_execution() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let result = tracker.execute_with_spinner("spinner_task", |spinner| {
        spinner.set_message("Working...");
        thread::sleep(Duration::from_millis(10));
        Ok("Success")
    });

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Success");
}

#[test]
fn test_progress_loop() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let pb = tracker.create_progress_bar("loop_test", 100);
    let mut progress_loop = ProgressLoop::new(pb, 10);

    // Simulate processing loop
    for _i in 0..25 {
        if !progress_loop.should_continue() {
            break;
        }
        progress_loop.update().expect("Progress update failed");
    }

    progress_loop.finish();
}

#[test]
fn test_progress_loop_cancellation() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let pb = tracker.create_progress_bar("cancellable_loop", 100);
    pb.cancel(); // Cancel immediately

    let mut progress_loop = ProgressLoop::new(pb, 10);

    // Should not continue if cancelled
    assert!(!progress_loop.should_continue());

    // Update should return error for cancelled operation
    let result = progress_loop.update();
    assert!(result.is_err());
}

#[test]
fn test_progress_info_eta_calculation() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let pb = tracker.create_progress_bar("eta_test", 100);

    // Initial state - no ETA available
    let progress = pb.get_progress();
    assert!(progress.eta.is_none());

    // Add some progress
    pb.set_position(25);
    thread::sleep(Duration::from_millis(10)); // Small delay to calculate rate

    let progress = pb.get_progress();
    assert_eq!(progress.current, 25);
    assert_eq!(progress.percentage, 25.0);
    // ETA might be available now, but depends on timing
}

#[test]
fn test_progress_cleanup_completed() {
    let config = create_test_config();
    let tracker = ProgressTracker::new(&config).expect("Failed to create progress tracker");

    let pb1 = tracker.create_progress_bar("completed_task", 100);
    let _pb2 = tracker.create_progress_bar("active_task", 100);

    // Finish one task
    pb1.finish();

    // Before cleanup
    let operations = tracker.get_active_operations();
    assert_eq!(operations.len(), 2);

    // After cleanup
    tracker.cleanup_completed();
    let operations = tracker.get_active_operations();
    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].name, "active_task");
}

#[test]
fn test_concurrent_progress_operations() {
    let config = create_test_config();
    let tracker = std::sync::Arc::new(
        ProgressTracker::new(&config).expect("Failed to create progress tracker")
    );

    let mut handles = vec![];

    // Spawn multiple threads with progress tracking
    for i in 0..5 {
        let tracker_clone = tracker.clone();
        let handle = std::thread::spawn(move || {
            let pb = tracker_clone.create_progress_bar(&format!("concurrent_task_{}", i), 100);

            for j in 0..100 {
                pb.inc();
                if j % 10 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }

            pb.finish();
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Check final state
    tracker.cleanup_completed();
    let operations = tracker.get_active_operations();
    assert_eq!(operations.len(), 0);
}