// consensus/core/validator/lifecycle.rs
use crate::types::*;
use std::collections::{BTreeMap, VecDeque, BinaryHeap};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::{
    distribution::{Normal, Exponential, Gamma, Continuous},
    statistics::Statistics,
};
use nalgebra::{DVector, DMatrix, SVD};

pub struct ValidatorLifecycleManager {
    activation_queue: Arc<RwLock<BinaryHeap<QueuedValidator>>>,
    exit_queue: Arc<RwLock<VecDeque<ExitingValidator>>>,
    jail_registry: Arc<RwLock<BTreeMap<ValidatorId, JailedValidator>>>,
    rehabilitation_engine: RehabilitationEngine,
    churn_optimizer: ChurnOptimizer,
    capacity_analyzer: NetworkCapacityAnalyzer,
    lifecycle_metrics: Arc<Mutex<LifecycleMetrics>>,
}

#[derive(Debug, Clone)]
pub struct QueuedValidator {
    pub validator: PendingValidator,
    pub submission_epoch: Epoch,
    pub priority_score: f64,
    pub technical_compliance: TechnicalComplianceScore,
    pub stake_commitment: StakeCommitment,
}

impl PartialOrd for QueuedValidator {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedValidator {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority_score
            .partial_cmp(&other.priority_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.submission_epoch.cmp(&other.submission_epoch))
    }
}

impl PartialEq for QueuedValidator {
    fn eq(&self, other: &Self) -> bool {
        self.validator.identity.id == other.validator.identity.id
    }
}

impl Eq for QueuedValidator {}

impl ValidatorLifecycleManager {
    pub async fn process_epoch_lifecycle_transitions(
        &self,
        current_epoch: Epoch,
        network_state: &NetworkState,
        economic_params: &EconomicParameters,
        security_metrics: &SecurityMetrics,
    ) -> Result<EpochLifecycleResult, LifecycleError> {
        let mut lifecycle_result = EpochLifecycleResult::new(current_epoch);

        // Phase 1: Calculate dynamic churn limits
        let (activation_churn, exit_churn) = self.calculate_dynamic_churn_limits(
            network_state, 
            economic_params, 
            security_metrics
        ).await?;

        // Phase 2: Process validator activations with priority scheduling
        let activation_results = self.process_priority_activations(
            current_epoch, 
            activation_churn, 
            network_state
        ).await?;
        lifecycle_result.activation_results = activation_results;

        // Phase 3: Process voluntary exits with fairness optimization
        let exit_results = self.process_fairness_exits(
            current_epoch, 
            exit_churn, 
            network_state
        ).await?;
        lifecycle_result.exit_results = exit_results;

        // Phase 4: Process forced exits and slashing events
        let forced_exit_results = self.process_forced_exits_and_slashing(
            current_epoch, 
            security_metrics
        ).await?;
        lifecycle_result.forced_exit_results = forced_exit_results;

        // Phase 5: Process jail releases and rehabilitation
        let rehabilitation_results = self.process_rehabilitation_transitions(
            current_epoch, 
            network_state
        ).await?;
        lifecycle_result.rehabilitation_results = rehabilitation_results;

        // Phase 6: Update network capacity and optimize queues
        let capacity_updates = self.optimize_network_capacity(
            &lifecycle_result, 
            network_state, 
            economic_params
        ).await?;
        lifecycle_result.capacity_updates = capacity_updates;

        // Phase 7: Calculate comprehensive lifecycle metrics
        lifecycle_result.lifecycle_metrics = self.calculate_comprehensive_lifecycle_metrics().await?;

        Ok(lifecycle_result)
    }

    pub async fn queue_validator_activation(
        &self,
        pending_validator: PendingValidator,
        current_epoch: Epoch,
        network_state: &NetworkState,
    ) -> Result<ActivationQueueResult, LifecycleError> {
        // Phase 1: Comprehensive eligibility validation
        let eligibility_validation = self.validate_activation_eligibility(
            &pending_validator, 
            current_epoch, 
            network_state
        ).await?;

        if !eligibility_validation.eligible {
            return Err(LifecycleError::ActivationEligibilityFailed(
                eligibility_validation.reasons
            ));
        }

        // Phase 2: Technical compliance assessment
        let technical_compliance = self.assess_technical_compliance(
            &pending_validator
        ).await?;

        if technical_compliance.overall_score < self.config.min_technical_score {
            return Err(LifecycleError::TechnicalRequirementsNotMet(
                technical_compliance.failed_checks
            ));
        }

        // Phase 3: Calculate priority score using multi-factor optimization
        let priority_score = self.calculate_activation_priority_score(
            &pending_validator,
            &technical_compliance,
            current_epoch,
            network_state
        ).await?;

        // Phase 4: Create queued validator with comprehensive metadata
        let queued_validator = QueuedValidator {
            validator: pending_validator.clone(),
            submission_epoch: current_epoch,
            priority_score,
            technical_compliance: technical_compliance.overall_score,
            stake_commitment: self.calculate_stake_commitment(&pending_validator).await?,
        };

        // Phase 5: Add to priority queue with concurrency control
        let mut queue = self.activation_queue.write().await;
        queue.push(queued_validator);

        // Phase 6: Calculate queue dynamics and estimated activation
        let queue_analysis = self.analyze_activation_queue(&queue, current_epoch).await?;
        let estimated_activation = self.calculate_estimated_activation_epoch(
            &queue_analysis, 
            current_epoch
        ).await?;

        Ok(ActivationQueueResult {
            validator_id: pending_validator.identity.id,
            queue_position: queue_analysis.estimated_position,
            estimated_activation_epoch: estimated_activation,
            submission_epoch: current_epoch,
            activation_eligibility_epoch: self.calculate_eligibility_epoch(current_epoch).await?,
            priority_score,
            queue_status: QueueStatus::Pending,
            technical_compliance_score: technical_compliance.overall_score,
            stake_commitment_level: self.assess_stake_commitment_level(&pending_validator).await?,
        })
    }

    async fn process_priority_activations(
        &self,
        current_epoch: Epoch,
        churn_limit: usize,
        network_state: &NetworkState,
    ) -> Result<Vec<ActivationResult>, LifecycleError> {
        let mut activations = Vec::with_capacity(churn_limit);
        let mut queue = self.activation_queue.write().await;
        let mut activated_count = 0;

        while activated_count < churn_limit && !queue.is_empty() {
            if let Some(queued_validator) = queue.pop() {
                // Comprehensive activation eligibility check
                if self.is_eligible_for_immediate_activation(
                    &queued_validator, 
                    current_epoch, 
                    network_state
                ).await? {
                    let activation_result = self.execute_validator_activation(
                        queued_validator, 
                        current_epoch
                    ).await?;
                    activations.push(activation_result);
                    activated_count += 1;
                } else {
                    // Re-queue with updated priority if not immediately eligible
                    let updated_priority = self.update_queued_validator_priority(
                        queued_validator, 
                        current_epoch
                    ).await?;
                    queue.push(updated_priority);
                }
            }
        }

        // Optimize queue ordering after processing
        self.optimize_activation_queue_ordering(&mut queue).await?;

        Ok(activations)
    }

    async fn execute_validator_activation(
        &self,
        queued_validator: QueuedValidator,
        current_epoch: Epoch,
    ) -> Result<ActivationResult, LifecycleError> {
        let pending_validator = queued_validator.validator;

        // Phase 1: Create comprehensive active validator state
        let active_validator = self.create_active_validator_state(
            pending_validator, 
            current_epoch
        ).await?;

        // Phase 2: Initialize advanced performance metrics
        let performance_metrics = self.initialize_advanced_performance_metrics(
            &active_validator
        ).await?;

        // Phase 3: Initialize time-lived state with historical context
        let time_lived_state = self.initialize_time_lived_state_with_history(
            &active_validator, 
            current_epoch
        ).await?;

        // Phase 4: Register with consensus subsystems
        self.register_with_consensus_subsystems(
            active_validator.clone()
        ).await?;

        // Phase 5: Update network-wide statistics and capacity
        self.update_network_activation_metrics(
            current_epoch, 
            &active_validator
        ).await?;

        Ok(ActivationResult {
            validator_id: active_validator.identity.id,
            activation_epoch: current_epoch,
            initial_stake: active_validator.stake_state.effective_stake,
            activation_status: ActivationStatus::Active,
            performance_metrics,
            time_lived_state,
            technical_compliance: queued_validator.technical_compliance,
            priority_score: queued_validator.priority_score,
            network_impact: self.assess_network_impact(&active_validator).await?,
        })
    }

    pub async fn initiate_voluntary_exit(
        &self,
        validator_id: ValidatorId,
        current_epoch: Epoch,
        exit_request: &ExitRequest,
        network_state: &NetworkState,
    ) -> Result<ExitQueueResult, LifecycleError> {
        // Phase 1: Comprehensive exit eligibility validation
        let exit_validation = self.validate_exit_eligibility(
            validator_id, 
            current_epoch, 
            exit_request, 
            network_state
        ).await?;

        if !exit_validation.eligible {
            return Err(LifecycleError::ExitEligibilityFailed(
                exit_validation.reasons
            ));
        }

        // Phase 2: Calculate exit penalties and economic impact
        let economic_analysis = self.analyze_exit_economic_impact(
            validator_id, 
            current_epoch, 
            exit_request
        ).await?;

        // Phase 3: Create exiting validator with comprehensive state
        let exiting_validator = self.create_exiting_validator_record(
            validator_id, 
            exit_request, 
            current_epoch, 
            &economic_analysis
        ).await?;

        // Phase 4: Calculate exit queue position with fairness considerations
        let queue_position = self.calculate_fair_exit_queue_position(
            validator_id, 
            &exiting_validator
        ).await?;

        let estimated_exit = self.estimate_exit_epoch_with_uncertainty(
            queue_position, 
            current_epoch, 
            network_state
        ).await?;

        // Phase 5: Add to exit queue with priority scheduling
        let mut exit_queue = self.exit_queue.write().await;
        exit_queue.push_back(exiting_validator);

        // Phase 6: Update validator state to exiting
        self.transition_validator_to_exiting_state(
            validator_id, 
            current_epoch, 
            exit_request
        ).await?;

        Ok(ExitQueueResult {
            validator_id,
            queue_position,
            estimated_exit_epoch: estimated_exit.expected_epoch,
            exit_initiation_epoch: current_epoch,
            withdrawable_epoch: self.calculate_withdrawable_epoch(estimated_exit.expected_epoch).await?,
            exit_penalty: economic_analysis.penalty_amount,
            economic_impact: economic_analysis.overall_impact,
            confidence_interval: estimated_exit.confidence_interval,
            queue_priority: self.determine_exit_priority(&economic_analysis).await?,
        })
    }

    async fn process_fairness_exits(
        &self,
        current_epoch: Epoch,
        churn_limit: usize,
        network_state: &NetworkState,
    ) -> Result<Vec<ExitResult>, LifecycleError> {
        let mut exits = Vec::with_capacity(churn_limit);
        let mut exit_queue = self.exit_queue.write().await;
        let mut processed_count = 0;

        while processed_count < churn_limit && !exit_queue.is_empty() {
            if let Some(exiting_validator) = exit_queue.pop_front() {
                if self.is_eligible_for_exit_processing(
                    &exiting_validator, 
                    current_epoch, 
                    network_state
                ).await? {
                    let exit_result = self.finalize_validator_exit(
                        exiting_validator, 
                        current_epoch
                    ).await?;
                    exits.push(exit_result);
                    processed_count += 1;
                } else {
                    // Return to queue with updated state
                    exit_queue.push_front(exiting_validator);
                    break;
                }
            }
        }

        // Optimize exit queue for fairness
        self.optimize_exit_queue_fairness(&mut exit_queue).await?;

        Ok(exits)
    }

    async fn finalize_validator_exit(
        &self,
        exiting_validator: ExitingValidator,
        current_epoch: Epoch,
    ) -> Result<ExitResult, LifecycleError> {
        // Phase 1: Process comprehensive stake withdrawal
        let withdrawal_processing = self.process_stake_withdrawal_optimization(
            &exiting_validator, 
            current_epoch
        ).await?;

        // Phase 2: Finalize validator state with historical preservation
        let state_finalization = self.finalize_validator_state_with_history(
            exiting_validator.validator_id, 
            current_epoch
        ).await?;

        // Phase 3: Update network exit statistics and capacity
        self.update_network_exit_metrics(
            current_epoch, 
            &exiting_validator
        ).await?;

        // Phase 4: Clean up validator records with archival
        self.archive_validator_records(
            exiting_validator.validator_id
        ).await?;

        Ok(ExitResult {
            validator_id: exiting_validator.validator_id,
            exit_epoch: current_epoch,
            withdrawn_stake: withdrawal_processing.withdrawn_amount,
            exit_penalty: withdrawal_processing.penalty_amount,
            final_balance: withdrawal_processing.final_balance,
            exit_type: exiting_validator.exit_type,
            withdrawable_epoch: self.calculate_final_withdrawable_epoch(current_epoch).await?,
            historical_preservation: state_finalization.historical_data_preserved,
            network_impact: self.assess_exit_network_impact(&exiting_validator).await?,
        })
    }

    pub async fn handle_forced_exit(
        &self,
        validator_id: ValidatorId,
        forced_exit_reason: ForcedExitReason,
        current_epoch: Epoch,
        security_context: &SecurityContext,
    ) -> Result<ForcedExitResult, LifecycleError> {
        // Phase 1: Validate forced exit conditions with security context
        let forced_exit_validation = self.validate_forced_exit_conditions(
            validator_id, 
            &forced_exit_reason, 
            security_context
        ).await?;

        if !forced_exit_validation.valid {
            return Err(LifecycleError::ForcedExitValidationFailed(
                forced_exit_validation.reasons
            ));
        }

        // Phase 2: Calculate comprehensive forced exit penalties
        let penalty_calculation = self.calculate_forced_exit_penalties(
            validator_id, 
            &forced_exit_reason, 
            security_context
        ).await?;

        // Phase 3: Create forced exit record with emergency priority
        let forced_exit = self.create_forced_exit_record(
            validator_id, 
            forced_exit_reason, 
            penalty_calculation, 
            current_epoch
        ).await?;

        // Phase 4: Immediate queue placement with highest priority
        let mut exit_queue = self.exit_queue.write().await;
        exit_queue.push_front(forced_exit);

        // Phase 5: Update validator status to forced exiting
        self.mark_validator_forced_exit(
            validator_id, 
            current_epoch, 
            &penalty_calculation
        ).await?;

        Ok(ForcedExitResult {
            validator_id,
            forced_exit_epoch: current_epoch,
            exit_reason: forced_exit_reason,
            penalty_amount: penalty_calculation.total_penalty,
            queue_priority: ExitPriority::Emergency,
            estimated_exit_epoch: current_epoch,
            security_implications: penalty_calculation.security_impact,
            network_protection: self.assess_network_protection_benefit(validator_id).await?,
        })
    }

    async fn calculate_dynamic_churn_limits(
        &self,
        network_state: &NetworkState,
        economic_params: &EconomicParameters,
        security_metrics: &SecurityMetrics,
    ) -> Result<(usize, usize), LifecycleError> {
        let base_activation_churn = 4;
        let base_exit_churn = 4;

        // Calculate activation churn limit with multiple factors
        let activation_churn = self.calculate_activation_churn_limit(
            network_state, 
            economic_params, 
            security_metrics
        ).await?;

        // Calculate exit churn limit with stability considerations
        let exit_churn = self.calculate_exit_churn_limit(
            network_state, 
            economic_params, 
            security_metrics
        ).await?;

        // Apply network stability constraints
        let constrained_activation = self.apply_stability_constraints(
            activation_churn, 
            network_state, 
            ChurnType::Activation
        ).await?;

        let constrained_exit = self.apply_stability_constraints(
            exit_churn, 
            network_state, 
            ChurnType::Exit
        ).await?;

        Ok((constrained_activation, constrained_exit))
    }

    async fn calculate_activation_churn_limit(
        &self,
        network_state: &NetworkState,
        economic_params: &EconomicParameters,
        security_metrics: &SecurityMetrics,
    ) -> Result<usize, LifecycleError> {
        let validator_count = network_state.active_validators;
        let network_capacity = network_state.network_capacity;
        
        // Base churn with logarithmic scaling
        let base_churn = 4;
        let size_scaling = (validator_count as f64).ln().max(1.0);
        
        // Security-based adjustment
        let security_factor = match security_metrics.overall_security_level {
            SecurityLevel::High => 0.8,
            SecurityLevel::Medium => 1.0,
            SecurityLevel::Low => 1.2,
        };
        
        // Economic-based adjustment
        let economic_factor = if economic_params.inflation_rate > economic_params.target_inflation_rate {
            0.9 // Reduce churn during high inflation
        } else {
            1.1 // Increase churn during low inflation
        };
        
        // Capacity-based adjustment
        let capacity_utilization = validator_count as f64 / network_capacity as f64;
        let capacity_factor = if capacity_utilization > 0.8 {
            0.7 // Reduce churn near capacity
        } else if capacity_utilization < 0.5 {
            1.3 // Increase churn with available capacity
        } else {
            1.0
        };
        
        let scaled_churn = (base_churn as f64 * size_scaling * security_factor * economic_factor * capacity_factor) as usize;
        
        // Apply bounds
        Ok(scaled_churn.max(2).min(64))
    }

    async fn calculate_exit_churn_limit(
        &self,
        network_state: &NetworkState,
        economic_params: &EconomicParameters,
        security_metrics: &SecurityMetrics,
    ) -> Result<usize, LifecycleError> {
        let validator_count = network_state.active_validators;
        let network_stability = network_state.network_stability;
        
        // Base churn with different scaling for exits
        let base_churn = 4;
        let size_scaling = (validator_count as f64 * 0.7).ln().max(1.0);
        
        // Stability-based adjustment
        let stability_factor = if network_stability > 0.8 {
            1.2 // Allow more exits during stable periods
        } else if network_stability < 0.5 {
            0.8 // Restrict exits during unstable periods
        } else {
            1.0
        };
        
        // Security-based adjustment (more conservative for exits)
        let security_factor = match security_metrics.overall_security_level {
            SecurityLevel::High => 1.1,
            SecurityLevel::Medium => 1.0,
            SecurityLevel::Low => 0.9, // Restrict exits during low security
        };
        
        let scaled_churn = (base_churn as f64 * size_scaling * stability_factor * security_factor) as usize;
        
        // Apply bounds (exits typically have lower limits than activations)
        Ok(scaled_churn.max(2).min(32))
    }
}

pub struct RehabilitationEngine {
    criteria_evaluator: MultiCriteriaEvaluator,
    performance_assessor: AdvancedPerformanceAssessor,
    stake_compliance_verifier: StakeComplianceVerifier,
    community_reputation_engine: CommunityReputationEngine,
    rehabilitation_optimizer: RehabilitationOptimizer,
}

impl RehabilitationEngine {
    pub async fn evaluate_rehabilitation_eligibility(
        &self,
        jailed_validator: &JailedValidator,
        current_epoch: Epoch,
        network_state: &NetworkState,
    ) -> Result<RehabilitationEvaluation, LifecycleError> {
        let mut evaluation = RehabilitationEvaluation::new(jailed_validator.validator.identity.id);

        // Criterion 1: Time served with good behavior
        evaluation.time_served_criteria = self.evaluate_time_served_criteria(
            jailed_validator, 
            current_epoch
        ).await?;

        // Criterion 2: Stake compliance and economic commitment
        evaluation.stake_compliance_criteria = self.evaluate_stake_compliance_criteria(
            jailed_validator
        ).await?;

        // Criterion 3: Behavioral improvement and pattern analysis
        evaluation.behavioral_improvement_criteria = self.evaluate_behavioral_improvement_criteria(
            jailed_validator
        ).await?;

        // Criterion 4: Technical capability and infrastructure
        evaluation.technical_capability_criteria = self.evaluate_technical_capability_criteria(
            jailed_validator
        ).await?;

        // Criterion 5: Community reputation and trust
        evaluation.community_reputation_criteria = self.evaluate_community_reputation_criteria(
            jailed_validator, 
            network_state
        ).await?;

        // Calculate overall rehabilitation score
        evaluation.overall_score = self.calculate_rehabilitation_score(&evaluation).await?;
        evaluation.eligible = evaluation.overall_score >= self.config.rehabilitation_threshold;

        // Determine rehabilitation pathway
        evaluation.rehabilitation_pathway = self.determine_rehabilitation_pathway(&evaluation).await?;

        Ok(evaluation)
    }

    pub async fn process_validator_rehabilitation(
        &self,
        jailed_validator: JailedValidator,
        current_epoch: Epoch,
        network_state: &NetworkState,
    ) -> Result<RehabilitationResult, LifecycleError> {
        // Phase 1: Comprehensive rehabilitation evaluation
        let rehabilitation_evaluation = self.evaluate_rehabilitation_eligibility(
            &jailed_validator, 
            current_epoch, 
            network_state
        ).await?;

        if rehabilitation_evaluation.eligible {
            // Phase 2: Execute successful rehabilitation
            let released_validator = self.execute_successful_rehabilitation(
                jailed_validator, 
                current_epoch, 
                &rehabilitation_evaluation
            ).await?;

            // Phase 3: Reactivate validator with probationary status
            let reactivation_result = self.reactivate_with_probation(
                released_validator, 
                current_epoch, 
                &rehabilitation_evaluation
            ).await?;

            Ok(RehabilitationResult {
                validator_id: jailed_validator.validator.identity.id,
                release_epoch: current_epoch,
                rehabilitation_status: RehabilitationStatus::Success,
                reactivation_result: Some(reactivation_result),
                probation_period: self.calculate_probation_period(&rehabilitation_evaluation).await?,
                rehabilitation_score: rehabilitation_evaluation.overall_score,
                pathway: rehabilitation_evaluation.rehabilitation_pathway,
            })
        } else {
            // Phase 2: Handle failed rehabilitation
            let extended_jail = self.handle_failed_rehabilitation(
                jailed_validator, 
                current_epoch, 
                &rehabilitation_evaluation
            ).await?;

            Ok(RehabilitationResult {
                validator_id: extended_jail.validator.identity.id,
                release_epoch: extended_jail.release_epoch,
                rehabilitation_status: RehabilitationStatus::Failed,
                reactivation_result: None,
                probation_period: self.calculate_extended_probation(&extended_jail).await?,
                rehabilitation_score: rehabilitation_evaluation.overall_score,
                pathway: RehabilitationPathway::ExtendedJail,
            })
        }
    }
}

pub struct ChurnOptimizer {
    stability_analyzer: NetworkStabilityAnalyzer,
    capacity_planner: NetworkCapacityPlanner,
    fairness_optimizer: FairnessOptimizer,
    predictive_model: ChurnPredictiveModel,
}

impl ChurnOptimizer {
    pub async fn optimize_churn_parameters(
        &self,
        network_state: &NetworkState,
        historical_churn: &[HistoricalChurnData],
        economic_indicators: &EconomicIndicators,
    ) -> Result<OptimizedChurnParameters, LifecycleError> {
        // Analyze historical churn patterns
        let churn_patterns = self.analyze_historical_churn_patterns(historical_churn).await?;
        
        // Predict future churn requirements
        let churn_predictions = self.predict_future_churn_requirements(
            network_state, 
            economic_indicators
        ).await?;
        
        // Calculate stability-optimized churn limits
        let stability_optimized = self.calculate_stability_optimized_churn(
            network_state, 
            &churn_patterns
        ).await?;
        
        // Apply fairness constraints
        let fairness_constrained = self.apply_fairness_constraints(
            stability_optimized, 
            network_state
        ).await?;
        
        // Final optimization with predictive adjustments
        let final_optimization = self.finalize_churn_optimization(
            fairness_constrained, 
            &churn_predictions
        ).await?;

        Ok(final_optimization)
    }
}