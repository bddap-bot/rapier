use crate::alloc_prelude::*;
use crate::dynamics::solver::{
    AnyJointConstraintMut, GenericJointConstraint, JointGenericExternalConstraintBuilder,
    JointGenericInternalConstraintBuilder,
};
use crate::dynamics::{JointGraphEdge, MultibodyJointSet, RigidBodySet};
use crate::math::DVector;
use parry::math::Real;

use crate::dynamics::solver::joint_constraint::generic_joint_constraint_builder::GenericJointConstraintBuilder;
use crate::dynamics::solver::joint_constraint::joint_constraint_builder::JointConstraintBuilder;
use crate::dynamics::solver::joint_constraint::joint_velocity_constraint::JointConstraint;
use {
    crate::dynamics::solver::joint_constraint::joint_constraint_builder::JointConstraintBuilderSimd,
    crate::math::{SIMD_WIDTH, SimdReal},
};

pub struct JointConstraintsSet {
    pub generic_jacobians: DVector,
    pub two_body_interactions: Vec<usize>,
    pub generic_two_body_interactions: Vec<usize>,

    pub generic_velocity_constraints: Vec<GenericJointConstraint>,
    pub velocity_constraints: Vec<JointConstraint<Real, 1>>,
    pub simd_velocity_constraints: Vec<JointConstraint<SimdReal, SIMD_WIDTH>>,

    pub generic_velocity_constraints_builder: Vec<GenericJointConstraintBuilder>,
    pub velocity_constraints_builder: Vec<JointConstraintBuilder>,
    pub simd_velocity_constraints_builder: Vec<JointConstraintBuilderSimd>,
}

impl JointConstraintsSet {
    pub fn new() -> Self {
        Self {
            generic_jacobians: DVector::zeros(0),
            two_body_interactions: vec![],
            generic_two_body_interactions: vec![],
            velocity_constraints: vec![],
            generic_velocity_constraints: vec![],
            simd_velocity_constraints: vec![],
            velocity_constraints_builder: vec![],
            generic_velocity_constraints_builder: vec![],
            simd_velocity_constraints_builder: vec![],
        }
    }

    // Returns the generic jacobians and a mutable iterator through all the constraints.
    pub fn iter_constraints_mut(
        &mut self,
    ) -> (&DVector, impl Iterator<Item = AnyJointConstraintMut<'_>>) {
        let jac = &self.generic_jacobians;
        let a = self
            .generic_velocity_constraints
            .iter_mut()
            .map(AnyJointConstraintMut::Generic);
        let b = self
            .velocity_constraints
            .iter_mut()
            .map(AnyJointConstraintMut::Rigid);
        let c = self
            .simd_velocity_constraints
            .iter_mut()
            .map(AnyJointConstraintMut::SimdRigid);
        (jac, a.chain(b).chain(c))
    }
}

impl JointConstraintsSet {
    /// Re-derives every constraint that touches a multibody (internal joint
    /// limits/motors, and external joints with a multibody side) from the
    /// multibodies' current positions and mass matrices, preserving the
    /// accumulated impulses so the impulse bounds keep their meaning.
    ///
    /// Called after the substep position integration so the bias-free
    /// stabilization solves apply impulses through the up-to-date inverse mass
    /// matrix: a multibody generalized impulse is momentum-neutral only
    /// through the mass matrix of the configuration it is applied at
    /// (bddap/rl#321). Rigid-rigid constraints are equal/opposite in maximal
    /// coordinates and never leak momentum, so they are left untouched.
    #[profiling::function]
    pub(crate) fn update_multibody_coupled(
        &mut self,
        params: &crate::dynamics::IntegrationParameters,
        multibodies: &MultibodyJointSet,
        solver_bodies: &crate::dynamics::solver::solver_body::SolverBodies,
    ) {
        // Also preserve the impulse bounds: re-deriving recomputes limit
        // activation from the post-integration position, and a limit that
        // deactivated mid-substep would get [0, 0] bounds — forcing the
        // stabilization solve to retract the impulse it already applied, i.e.
        // an elastic bounce where the stop was inelastic. The re-derivation is
        // only meant to refresh the jacobians and mass matrix.
        let impulses: Vec<(Real, [Real; 2])> = self
            .generic_velocity_constraints
            .iter()
            .map(|c| (c.impulse, c.impulse_bounds))
            .collect();

        for builder in &mut self.generic_velocity_constraints_builder {
            match builder {
                GenericJointConstraintBuilder::External(builder) => {
                    builder.update(
                        params,
                        multibodies,
                        solver_bodies,
                        &mut self.generic_jacobians,
                        &mut self.generic_velocity_constraints,
                    );
                }
                GenericJointConstraintBuilder::Internal(builder) => {
                    builder.update(
                        params,
                        multibodies,
                        &mut self.generic_jacobians,
                        &mut self.generic_velocity_constraints,
                    );
                }
                GenericJointConstraintBuilder::Empty => {}
            }
        }

        for (c, (impulse, bounds)) in self
            .generic_velocity_constraints
            .iter_mut()
            .zip(impulses.into_iter())
        {
            c.impulse = impulse;
            c.impulse_bounds = bounds;
        }
    }

    pub(crate) fn compute_generic_joint_constraints(
        &mut self,
        island_bodies: &[crate::dynamics::RigidBodyHandle],
        bodies: &RigidBodySet,
        multibodies: &MultibodyJointSet,
        joints_all: &[JointGraphEdge],
        j_id: &mut usize,
    ) {
        // Count the internal and external constraints builder.
        let num_external_constraint_builders = self.generic_two_body_interactions.len();
        let mut num_internal_constraint_builders = 0;
        for handle in island_bodies {
            if let Some(link_id) = multibodies.rigid_body_link(*handle) {
                if JointGenericInternalConstraintBuilder::num_constraints(multibodies, link_id) > 0
                {
                    num_internal_constraint_builders += 1;
                }
            }
        }
        let total_num_builders =
            num_external_constraint_builders + num_internal_constraint_builders;

        // Preallocate builders buffer.
        self.generic_velocity_constraints_builder
            .resize(total_num_builders, GenericJointConstraintBuilder::Empty);

        // Generate external constraints builders.
        let mut num_constraints = 0;
        for (joint_i, builder) in self
            .generic_two_body_interactions
            .iter()
            .zip(self.generic_velocity_constraints_builder.iter_mut())
        {
            let joint = &joints_all[*joint_i].weight;
            JointGenericExternalConstraintBuilder::generate(
                *joint_i,
                joint,
                bodies,
                multibodies,
                builder,
                j_id,
                &mut self.generic_jacobians,
                &mut num_constraints,
            );
        }

        // Generate internal constraints builder. They are indexed after the
        let mut curr_builder = self.generic_two_body_interactions.len();
        for handle in island_bodies {
            if curr_builder >= self.generic_velocity_constraints_builder.len() {
                break; // No more builder need to be generated.
            }

            if let Some(link_id) = multibodies.rigid_body_link(*handle) {
                let prev_num_constraints = num_constraints;
                JointGenericInternalConstraintBuilder::generate(
                    multibodies,
                    link_id,
                    &mut self.generic_velocity_constraints_builder[curr_builder],
                    j_id,
                    &mut self.generic_jacobians,
                    &mut num_constraints,
                );
                if num_constraints != prev_num_constraints {
                    curr_builder += 1;
                }
            }
        }

        // Resize constraints buffer now that we know the total count.
        self.generic_velocity_constraints
            .resize(num_constraints, GenericJointConstraint::invalid());
    }

    pub fn writeback_impulses(&mut self, joints_all: &mut [JointGraphEdge]) {
        let (_, constraints) = self.iter_constraints_mut();
        for mut c in constraints {
            c.writeback_impulses(joints_all);
        }
    }
}
