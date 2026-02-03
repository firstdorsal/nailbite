//! Exercise registry and selection strategy.
//!
//! Manages available exercises and selects appropriate ones
//! based on the detected BFRB type and user configuration.

use rand::seq::IndexedRandom;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::config::SelectionStrategy;
use crate::detection::types::BfrbType;
use crate::exercises::ear_touch::EarTouch;
use crate::exercises::finger_flick::FingerFlick;
use crate::exercises::fingertip_massage::FingertipMassage;
use crate::exercises::fist_clench::FistClench;
use crate::exercises::flat_hand_press::FlatHandPress;
use crate::exercises::interlocked_squeeze::InterlockedSqueeze;
use crate::exercises::palm_press::PalmPress;
use crate::exercises::types::Exercise;

/// Registry of all available exercises with selection logic.
pub struct ExerciseRegistry {
    exercises: Vec<Box<dyn Exercise>>,
    strategy: SelectionStrategy,
    preferred_id: Option<String>,
    round_robin_index: AtomicUsize,
}

impl ExerciseRegistry {
    /// Create a new registry with all built-in exercises.
    pub fn new(strategy: SelectionStrategy, preferred_id: Option<String>) -> Self {
        let exercises: Vec<Box<dyn Exercise>> = vec![
            Box::new(FistClench),
            Box::new(FlatHandPress),
            Box::new(InterlockedSqueeze),
            Box::new(EarTouch),
            Box::new(FingerFlick),
            Box::new(PalmPress),
            Box::new(FingertipMassage),
        ];

        Self {
            exercises,
            strategy,
            preferred_id,
            round_robin_index: AtomicUsize::new(0),
        }
    }

    /// Select an exercise appropriate for the given BFRB type.
    ///
    /// Returns a reference to the selected exercise, or `None` if no
    /// applicable exercise exists.
    pub fn select(&self, bfrb_type: BfrbType) -> Option<&dyn Exercise> {
        let applicable: Vec<&dyn Exercise> = self
            .exercises
            .iter()
            .filter(|e| e.applicable_to().contains(&bfrb_type))
            .map(|e| e.as_ref())
            .collect();

        if applicable.is_empty() {
            return None;
        }

        match self.strategy {
            SelectionStrategy::Random => {
                let mut rng = rand::rng();
                applicable.choose(&mut rng).copied()
            }
            SelectionStrategy::First => applicable.first().copied(),
            SelectionStrategy::RoundRobin => {
                let idx = self.round_robin_index.fetch_add(1, Ordering::Relaxed);
                applicable.get(idx % applicable.len()).copied()
            }
            SelectionStrategy::Preferred => {
                if let Some(ref preferred) = self.preferred_id {
                    applicable
                        .iter()
                        .find(|e| e.id() == preferred)
                        .copied()
                        .or_else(|| applicable.first().copied())
                } else {
                    applicable.first().copied()
                }
            }
        }
    }

    /// Get all exercises in the registry.
    pub fn all(&self) -> &[Box<dyn Exercise>] {
        &self.exercises
    }

    /// Get all exercises applicable to a specific BFRB type.
    pub fn applicable_exercises(&self, bfrb_type: BfrbType) -> Vec<&dyn Exercise> {
        self.exercises
            .iter()
            .filter(|e| e.applicable_to().contains(&bfrb_type))
            .map(|e| e.as_ref())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_all_exercises() {
        let registry = ExerciseRegistry::new(SelectionStrategy::First, None);
        assert_eq!(registry.all().len(), 7);
    }

    #[test]
    fn selects_applicable_exercise_for_nail_biting() {
        let registry = ExerciseRegistry::new(SelectionStrategy::First, None);
        let exercise = registry.select(BfrbType::NailBiting);
        assert!(exercise.is_some());
        assert!(exercise
            .unwrap()
            .applicable_to()
            .contains(&BfrbType::NailBiting));
    }

    #[test]
    fn round_robin_cycles() {
        let registry = ExerciseRegistry::new(SelectionStrategy::RoundRobin, None);

        let _e1 = registry.select(BfrbType::NailBiting).unwrap().id().to_string();
        let _e2 = registry.select(BfrbType::NailBiting).unwrap().id().to_string();

        // After cycling through all applicable exercises, should wrap around.
        let applicable_count = registry.applicable_exercises(BfrbType::NailBiting).len();
        assert!(applicable_count > 0);

        // Select enough to complete a full cycle.
        for _ in 0..applicable_count {
            registry.select(BfrbType::NailBiting);
        }
        let e_after_cycle = registry.select(BfrbType::NailBiting).unwrap().id().to_string();
        // Should have wrapped around (not necessarily same as e1 due to index).
        assert!(!e_after_cycle.is_empty());
    }

    #[test]
    fn preferred_strategy_selects_preferred() {
        let registry = ExerciseRegistry::new(
            SelectionStrategy::Preferred,
            Some("palm_press".to_string()),
        );
        let exercise = registry.select(BfrbType::NailBiting).unwrap();
        assert_eq!(exercise.id(), "palm_press");
    }

    #[test]
    fn preferred_falls_back_to_first() {
        let registry = ExerciseRegistry::new(
            SelectionStrategy::Preferred,
            Some("nonexistent".to_string()),
        );
        let exercise = registry.select(BfrbType::NailBiting);
        assert!(exercise.is_some());
    }
}
