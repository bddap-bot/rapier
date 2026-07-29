//! Linear-momentum conservation tests for the multibody solver (bddap/rl#321).
//!
//! Joint motors and limits are internal forces: for a floating multibody with
//! no contacts and no gravity, the total linear momentum of the links must
//! stay zero no matter how the joints are driven.

use crate::prelude::*;

struct World {
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd: CCDSolver,
    params: IntegrationParameters,
}

impl World {
    fn new() -> Self {
        let mut params = IntegrationParameters::default();
        params.dt = 1.0 / 64.0;
        Self {
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            params,
        }
    }

    fn step(&mut self) {
        self.step_with_gravity(Vector::ZERO);
    }

    fn step_with_gravity(&mut self, gravity: Vector) {
        self.pipeline.step(
            gravity,
            &self.params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
    }

    fn momentum(&self) -> Vector {
        self.bodies
            .iter()
            .map(|(_, rb)| rb.vels.linvel * rb.mprops.mass())
            .sum()
    }
}

fn free_body(world: &mut World, pos: Vector) -> RigidBodyHandle {
    let mprops = MassProperties::new(Vector::ZERO, 1.0, Vector::new(0.05, 0.05, 0.05));
    world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(pos)
            .additional_mass_properties(mprops),
    )
}

/// Drives a 2-link floating multibody's revolute joint with a square-wave
/// velocity motor and reports the peak |p| over the run.
fn thrash_peak_momentum(limits: [Real; 2]) -> Real {
    let mut world = World::new();
    let a = free_body(&mut world, Vector::new(0.0, 0.0, 0.0));
    let b = free_body(&mut world, Vector::new(1.0, 0.0, 0.0));

    let joint = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
        .local_anchor1(Vector::new(0.5, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.5, 0.0, 0.0))
        .limits(limits)
        .motor_max_force(200.0)
        .motor_velocity(8.0, 0.0);
    let handle = world.multibody_joints.insert(a, b, joint, true).unwrap();

    let mut peak: Real = 0.0;
    for i in 0..256 {
        // Square-wave thrash: slam the joint into alternating limits.
        let dir = if (i / 24) % 2 == 0 { 8.0 } else { -8.0 };
        let (mb, link_id) = world.multibody_joints.get_mut(handle).unwrap();
        mb.link_mut(link_id)
            .unwrap()
            .joint
            .data
            .set_motor_velocity(JointAxis::AngX, dir, 0.0);

        world.step();
        peak = peak.max(world.momentum().length());
    }
    peak
}

/// No motor, no limit hits, no gravity: spin the joint dof and coast.
/// Exercises only the bias/coriolis force path and the integrator.
#[test]
fn airborne_coasting_conserves_momentum() {
    let mut world = World::new();
    let a = free_body(&mut world, Vector::new(0.0, 0.0, 0.0));
    let b = free_body(&mut world, Vector::new(1.0, 0.0, 0.0));

    let joint = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
        .local_anchor1(Vector::new(0.5, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.5, 0.0, 0.0));
    let handle = world.multibody_joints.insert(a, b, joint, true).unwrap();

    // Step once so the multibody assembles, then set the joint dof velocity.
    world.step();
    {
        let (mb, _) = world.multibody_joints.get_mut(handle).unwrap();
        let ndofs = mb.ndofs();
        let mut vels = mb.generalized_velocity_mut();
        vels[ndofs - 1] = 8.0; // the revolute dof (root free joint occupies the first 6)
    }

    // The contrived initial state has nonzero total momentum; physical
    // evolution must preserve it exactly, so the metric is drift from p(0).
    world.step();
    let p0 = world.momentum();
    let mut peak: Real = 0.0;
    for _ in 0..256 {
        world.step();
        peak = peak.max((world.momentum() - p0).length());
    }
    println!("coasting peak |dp| = {peak}");
    assert!(peak < 1.0e-3, "coasting peak |dp| = {peak}");
}

#[test]
fn airborne_motor_thrash_wide_limits_conserves_momentum() {
    let peak = thrash_peak_momentum([-100.0, 100.0]);
    println!("wide-limit peak |p| = {peak}");
    assert!(peak < 1.0e-3, "wide-limit peak |p| = {peak}");
}

#[test]
fn airborne_motor_thrash_hard_limits_conserves_momentum() {
    let peak = thrash_peak_momentum([-0.3, 0.3]);
    println!("hard-limit peak |p| = {peak}");
    assert!(peak < 1.0e-3, "hard-limit peak |p| = {peak}");
}

/// dt-scaling probe: same 4-second coast at dt = 1/64, 1/128, 1/256.
/// Discretization error shrinks with dt; a structural bug does not.
#[test]
fn coasting_drift_dt_scaling() {
    for (dt, steps) in [(1.0 / 64.0, 256), (1.0 / 128.0, 512), (1.0 / 256.0, 1024)] {
        let mut world = World::new();
        world.params.dt = dt;
        let a = free_body(&mut world, Vector::new(0.0, 0.0, 0.0));
        let b = free_body(&mut world, Vector::new(1.0, 0.0, 0.0));
        let joint = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
            .local_anchor1(Vector::new(0.5, 0.0, 0.0))
            .local_anchor2(Vector::new(-0.5, 0.0, 0.0));
        let handle = world.multibody_joints.insert(a, b, joint, true).unwrap();
        world.step();
        {
            let (mb, _) = world.multibody_joints.get_mut(handle).unwrap();
            let ndofs = mb.ndofs();
            let mut vels = mb.generalized_velocity_mut();
            vels[ndofs - 1] = 8.0;
        }
        world.step();
        let p0 = world.momentum();
        let mut peak: Real = 0.0;
        for _ in 0..steps {
            world.step();
            peak = peak.max((world.momentum() - p0).length());
        }
        println!("dt = {dt}: coasting peak |dp| = {peak}");
        assert!(peak < 1.0e-3, "dt = {dt}: coasting peak |dp| = {peak}");
    }
}

/// With gravity on and the joint thrashing, the COM must obey Δp = m·g·Δt
/// exactly — the ledger must credit external forces, not suppress them.
#[test]
fn airborne_thrash_under_gravity_obeys_mg_dt() {
    let gravity = Vector::new(0.0, -9.81, 0.0);
    let mut world = World::new();
    let a = free_body(&mut world, Vector::new(0.0, 0.0, 0.0));
    let b = free_body(&mut world, Vector::new(1.0, 0.0, 0.0));
    let joint = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
        .local_anchor1(Vector::new(0.5, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.5, 0.0, 0.0))
        .limits([-0.3, 0.3])
        .motor_max_force(200.0)
        .motor_velocity(8.0, 0.0);
    let handle = world.multibody_joints.insert(a, b, joint, true).unwrap();

    let mut worst: Real = 0.0;
    for i in 0..256 {
        let dir = if (i / 24) % 2 == 0 { 8.0 } else { -8.0 };
        let (mb, link_id) = world.multibody_joints.get_mut(handle).unwrap();
        mb.link_mut(link_id)
            .unwrap()
            .joint
            .data
            .set_motor_velocity(JointAxis::AngX, dir, 0.0);
        world.step_with_gravity(gravity);
        let t = world.params.dt * (i + 1) as Real;
        let expected = gravity * 2.0 * t; // m_total = 2
        worst = worst.max((world.momentum() - expected).length());
    }
    println!("gravity thrash worst |p - m·g·t| = {worst}");
    assert!(worst < 1.0e-3, "worst |p - m·g·t| = {worst}");
}

/// A multibody dropped onto ground must stay bounded and keep interacting with
/// the contact normally. The base solver never brings this scenario fully to
/// rest (multibody + soft-contact jitter predates the momentum fix), so this
/// guards against explosion-class regressions, not perfect rest.
#[test]
fn multibody_on_ground_stays_bounded() {
    let gravity = Vector::new(0.0, -9.81, 0.0);
    let mut world = World::new();

    let ground = world
        .bodies
        .insert(RigidBodyBuilder::fixed().translation(Vector::new(0.0, -1.0, 0.0)));
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 1.0, 50.0),
        ground,
        &mut world.bodies,
    );

    let a = free_body(&mut world, Vector::new(0.0, 1.0, 0.0));
    let b = free_body(&mut world, Vector::new(1.0, 1.0, 0.0));
    for h in [a, b] {
        world.colliders.insert_with_parent(
            ColliderBuilder::ball(0.2).density(0.0),
            h,
            &mut world.bodies,
        );
    }
    let joint = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
        .local_anchor1(Vector::new(0.5, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.5, 0.0, 0.0));
    world.multibody_joints.insert(a, b, joint, true).unwrap();

    let mut sum_p = 0.0;
    let mut n = 0;
    for i in 0..512 {
        world.step_with_gravity(gravity);
        let ya = world.bodies[a].translation().y;
        assert!(ya.is_finite() && (-0.5..2.0).contains(&ya), "ya = {ya}");
        if i >= 256 {
            sum_p += world.momentum().length();
            n += 1;
        }
    }
    let mean_p = sum_p / n as Real;
    println!("windowed mean |p| = {mean_p:.4}");
    // Baseline (pre-fix) measures ~3.2 here; the momentum fix ~3.6.
    assert!(mean_p < 8.0, "windowed mean |p| = {mean_p}");
}

/// Inserting a multibody joint between bodies that already stepped (no pending
/// change flags) must still assemble the multibody's kinematics: the pipeline's
/// gated kinematics refresh has to catch structural mutations, not just
/// rigid-body change flags.
#[test]
fn multibody_joint_insertion_after_settled_step_assembles() {
    let mut world = World::new();
    let a = free_body(&mut world, Vector::new(0.0, 0.0, 0.0));
    let b = free_body(&mut world, Vector::new(1.0, 0.0, 0.0));

    // Step a few times so all change flags are cleared.
    for _ in 0..3 {
        world.step();
    }

    let joint = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
        .local_anchor1(Vector::new(0.5, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.5, 0.0, 0.0));
    world.multibody_joints.insert(a, b, joint, true).unwrap();

    // Used to panic (zero-sized body jacobians reaching the solver) when the
    // kinematics refresh was gated on change flags alone.
    for _ in 0..8 {
        world.step();
    }
    assert!(world.bodies[a].translation().is_finite());

    // And removal must likewise re-assemble the splits.
    let handle = world
        .multibody_joints
        .rigid_body_link(b)
        .copied()
        .map(|link| link.multibody);
    assert!(handle.is_some());
    let joint_handle = world.multibody_joints.attached_joints(a).next().unwrap().2;
    world.multibody_joints.remove(joint_handle, true);
    for _ in 0..8 {
        world.step();
    }
    assert!(world.bodies[a].translation().is_finite());
}
