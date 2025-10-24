// src/core/scoring/mod.rs

pub mod engine;
pub mod time_lived;
pub mod stochastic;
pub mod weights_manager;

pub fn init_scoring_engine() {
    println!("Scoring engine initialized.");
}